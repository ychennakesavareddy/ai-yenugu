import os
import time
import uuid
import json
import logging
import secrets
import datetime
import traceback
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import jwt
from jwt.exceptions import InvalidTokenError
import zlib

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_API_URL = "https://api.cohere.ai/v1/chat"
DEFAULT_CHAT_TITLE = "New Chat"
MAX_CHAT_MESSAGE_LENGTH = 50000
MAX_CHAT_TITLE_LENGTH = 100
MAX_RESPONSE_TOKENS = 4000
THREAD_POOL_SIZE = 4
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 30
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Initialize thread pool
executor = ThreadPoolExecutor(
    max_workers=THREAD_POOL_SIZE,
    thread_name_prefix='api_worker'
)

# Application Configuration
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config.update({
    'SECRET_KEY': os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32)),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'CHAT_HISTORY_LIMIT': 100,
    'JSONIFY_PRETTYPRINT_REGULAR': False,
    'JSON_SORT_KEYS': False,
    'JSONIFY_MIMETYPE': 'application/json',
})

# Initialize CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", FRONTEND_URL],
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "max_age": 600
    }
})

# Rate Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    strategy="fixed-window",
    default_limits=["500 per day", "100 per hour"],
    headers_enabled=True
)

# Error handlers
@app.errorhandler(RateLimitExceeded)
def handle_rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

# JWT Utilities
def create_jwt_token(user_id, email):
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXPIRE_MINUTES),
        "iat": datetime.datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except InvalidTokenError:
        return None

# Utility functions
def validate_chat_message(message):
    if not message or not isinstance(message, str):
        raise ValueError("Message must be a non-empty string")
    message = message.strip()
    if len(message) > MAX_CHAT_MESSAGE_LENGTH:
        raise ValueError(f"Message too long (max {MAX_CHAT_MESSAGE_LENGTH} characters)")
    return message

def validate_chat_title(title):
    if not title or not isinstance(title, str):
        raise ValueError("Title must be a non-empty string")
    title = title.strip()
    if len(title) > MAX_CHAT_TITLE_LENGTH:
        raise ValueError(f"Title too long (max {MAX_CHAT_TITLE_LENGTH} characters)")
    return title

def handle_api_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error: {str(e)}")
            return jsonify({"error": "External service unavailable"}), 503
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "An unexpected error occurred"}), 500
    return decorated_function

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def generate_cohere_response(message, chat_history=None):
    if not COHERE_API_KEY:
        raise ValueError("Cohere API key is not configured.")

    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
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
            for msg in chat_history[-10:]
            if msg["sender"] in ["user", "ai"]
        ]

    try:
        start_time = time.time()
        response = requests.post(
            COHERE_API_URL,
            headers=headers,
            json=data,
            timeout=(10, 60)
        )

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 5))
            time.sleep(retry_after)
            response.raise_for_status()

        response.raise_for_status()

        response_data = response.json()
        if "text" not in response_data:
            raise ValueError("Unexpected response format from Cohere API")

        logger.info(f"Cohere API call completed in {time.time() - start_time:.2f}s")
        return response_data["text"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Cohere API request failed: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode Cohere API response: {str(e)}")
        raise ValueError("Invalid response from Cohere API")
    except Exception as e:
        logger.error(f"Unexpected error in Cohere API call: {str(e)}")
        raise

# Routes
@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "cohere": bool(COHERE_API_KEY),
        }
    })

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")
@handle_api_errors
def chat():
    data = request.json
    try:
        message = validate_chat_message(data.get('message'))
        chat_id = data.get('chat_id')
        messages = data.get('messages', [])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    user_message = {
        "id": str(uuid.uuid4()),
        "content": message,
        "sender": "user",
        "timestamp": datetime.datetime.now().isoformat(),
        "error": False
    }

    messages.append(user_message)

    try:
        chat_history = messages[-10:]

        def generate_response():
            try:
                return generate_cohere_response(message, chat_history)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return None

        future = executor.submit(generate_response)
        ai_response = future.result(timeout=60)

        if not ai_response:
            raise Exception("Failed to generate response")

        ai_message = {
            "id": str(uuid.uuid4()),
            "content": ai_response,
            "sender": "ai",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": False
        }

        messages.append(ai_message)

        title = message[:MAX_CHAT_TITLE_LENGTH] + "..." if len(message) > MAX_CHAT_TITLE_LENGTH else message

        return jsonify({
            "response": ai_response,
            "chat_id": chat_id or str(uuid.uuid4()),
            "message": ai_message,
            "title": title,
            "messages": messages
        })

    except Exception as e:
        error_message = {
            "id": str(uuid.uuid4()),
            "content": f"Sorry, I couldn't process your request. Please try again later.",
            "sender": "system",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": True
        }
        messages.append(error_message)

        return jsonify({
            "error": str(e),
            "chat_id": chat_id,
            "message": error_message,
            "messages": messages
        }), 500

@app.route('/api/auth/signup', methods=['POST'])
@limiter.limit("10 per minute")
@handle_api_errors
def signup():
    data = request.json
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400

    # In a real app, you would:
    # 1. Validate the email and password
    # 2. Hash the password
    # 3. Store the user in a database
    # For this demo, we'll just create a JWT token
    
    token = create_jwt_token(data['email'], data['email'])
    return jsonify({
        "token": token,
        "user": {
            "email": data['email'],
            "name": data.get('name', data['email'].split('@')[0])
        }
    })

@app.route('/api/auth/signin', methods=['POST'])
@limiter.limit("10 per minute")
@handle_api_errors
def signin():
    data = request.json
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400

    # In a real app, you would:
    # 1. Verify the email exists
    # 2. Verify the password hash matches
    # For this demo, we'll just create a JWT token
    
    token = create_jwt_token(data['email'], data['email'])
    return jsonify({
        "token": token,
        "user": {
            "email": data['email'],
            "name": data.get('name', data['email'].split('@')[0])
        }
    })

@app.route('/api/auth/validate', methods=['GET'])
@handle_api_errors
def validate_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Invalid authorization header"}), 401

    token = auth_header.split(' ')[1]
    payload = verify_jwt_token(token)
    if not payload:
        return jsonify({"error": "Invalid or expired token"}), 401

    return jsonify({
        "valid": True,
        "user": {
            "email": payload['email'],
            "name": payload['email'].split('@')[0]
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)