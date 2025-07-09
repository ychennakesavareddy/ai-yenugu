import os
import time
import uuid
import json
import logging
import hashlib
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import jwt
from jwt.exceptions import InvalidTokenError
import bcrypt
import redis

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_URL = "https://api.cohere.ai/v1/chat"
MAX_CHAT_MESSAGE_LENGTH = 50000
MAX_RESPONSE_TOKENS = 4000
THREAD_POOL_SIZE = 8  # Increased for better concurrency
JWT_SECRET = os.getenv("JWT_SECRET", os.urandom(32).hex())
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 1440
PASSWORD_HASH_ROUNDS = 12

# Initialize Redis for rate limiting storage
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", ""),
    decode_responses=True
)

# Initialize thread pool with more workers
executor = ThreadPoolExecutor(
    max_workers=THREAD_POOL_SIZE,
    thread_name_prefix='api_worker'
)

# Application Configuration
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())
app.config.update({
    'JSONIFY_PRETTYPRINT_REGULAR': False,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
})

# Initialize CORS with preflight options
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv("ALLOWED_ORIGINS", "").split(","),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 600
    }
})

# Rate Limiter with Redis storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}",
    strategy="fixed-window",
    default_limits=["200 per day", "50 per hour"],
    headers_enabled=True
)

# Database simulation (replace with real DB in production)
users_db = {}
chats_db = {}

# Utility functions with caching
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(PASSWORD_HASH_ROUNDS)).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_jwt_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": time.time() + JWT_EXPIRE_MINUTES * 60,
        "iat": time.time(),
        "jti": str(uuid.uuid4())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except InvalidTokenError:
        return None

# Optimized Cohere API call with caching
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def generate_cohere_response(message: str, chat_history: list = None) -> str:
    cache_key = f"cohere:{hashlib.md5((message + str(chat_history)).encode()).hexdigest()}"
    cached_response = redis_client.get(cache_key)
    
    if cached_response:
        return cached_response

    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "message": message,
        "model": "command",
        "temperature": 0.7,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "stream": False
    }

    if chat_history:
        data["chat_history"] = [
            {"role": "user" if msg["sender"] == "user" else "chatbot", "message": msg["content"]}
            for msg in chat_history[-5:]  # Only last 5 messages for context
        ]

    try:
        start_time = time.time()
        response = requests.post(
            COHERE_API_URL,
            headers=headers,
            json=data,
            timeout=(3, 30)  # Shorter timeouts for better failover
        )
        response.raise_for_status()
        result = response.json().get("text", "")
        
        # Cache successful responses for 5 minutes
        redis_client.setex(cache_key, 300, result)
        logger.info(f"Cohere API call completed in {time.time() - start_time:.3f}s")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Cohere API error: {str(e)}")
        raise

# Authentication middleware
def authenticate_request():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    return verify_jwt_token(auth_header.split(' ')[1])

# Routes
@app.route('/api/health', methods=['GET'])
@limiter.limit("10 per minute")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "endpoints": {
            "signin": "/api/auth/signin",
            "signup": "/api/auth/signup",
            "chat": "/api/chat"
        }
    })

@app.route('/api/auth/signin', methods=['POST'])
@limiter.limit("10 per minute")
def signin():
    try:
        data = request.get_json()
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"error": "Email and password required"}), 400

        email = data['email'].lower()
        password = data['password']
        user = users_db.get(email)

        if not user or not verify_password(password, user["password"]):
            return jsonify({"error": "Invalid credentials"}), 401

        token = create_jwt_token(user["id"], email)
        return jsonify({
            "token": token,
            "user": {
                "id": user["id"],
                "email": email,
                "name": user["name"]
            }
        }), 200

    except Exception as e:
        logger.error(f"Signin error: {str(e)}")
        return jsonify({"error": "Authentication failed"}), 500

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")
def chat():
    auth = authenticate_request()
    if not auth:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message or len(message) > MAX_CHAT_MESSAGE_LENGTH:
            return jsonify({"error": "Invalid message"}), 400

        chat_id = data.get('chat_id', str(uuid.uuid4()))
        user_id = auth["sub"]

        if chat_id not in chats_db:
            chats_db[chat_id] = {
                "user_id": user_id,
                "messages": [],
                "created_at": time.time(),
                "title": message[:50] + "..." if len(message) > 50 else message
            }

        user_message = {
            "id": str(uuid.uuid4()),
            "content": message,
            "sender": "user",
            "timestamp": time.time()
        }

        chats_db[chat_id]["messages"].append(user_message)

        # Generate response asynchronously
        future = executor.submit(
            generate_cohere_response,
            message,
            chats_db[chat_id]["messages"]
        )
        ai_response = future.result(timeout=30)  # Timeout after 30 seconds

        ai_message = {
            "id": str(uuid.uuid4()),
            "content": ai_response,
            "sender": "ai",
            "timestamp": time.time()
        }

        chats_db[chat_id]["messages"].append(ai_message)

        return jsonify({
            "response": ai_response,
            "chat_id": chat_id,
            "title": chats_db[chat_id]["title"],
            "messages": chats_db[chat_id]["messages"]
        }), 200

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Failed to process message"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)