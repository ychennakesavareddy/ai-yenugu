"""
AI Yenugu - Complete Fixed Version
Flask application with Redis session storage and Google Drive integration
"""

import os
import io
import time
import uuid
import json
import logging
import secrets
import mimetypes
import datetime
import traceback
from functools import wraps
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, session, Response, send_file
from flask_cors import CORS
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis

# =============================================
# INITIALIZATION AND CONFIGURATION
# =============================================

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# =============================================
# CONSTANTS
# =============================================

SESSION_COOKIE_NAME = 'ai_yenugu_session'
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://ai-yenugu.onrender.com/api/drive-callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://ai-yenugu.netlify.app")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PROFILE_FILENAME = "user_profile.json"
AVATAR_FILENAME_PREFIX = "user_avatar"
CHATS_FOLDER_NAME = "AI Chat Storage"
COHERE_API_URL = "https://api.cohere.ai/v1/chat"
DEFAULT_CHAT_TITLE = "New Chat"
MAX_CHAT_MESSAGE_LENGTH = 5000
MAX_CHAT_TITLE_LENGTH = 100

# =============================================
# LOGGING CONFIGURATION
# =============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================
# REDIS CONFIGURATION
# =============================================

redis_conn = redis.Redis(
    host='redis-14356.c252.ap-southeast-1-1.ec2.redns.redis-cloud.com',
    port=14356,
    username="default",
    password="6T18RXJsymUjiqNNkEZgVucArLAyi080",
    decode_responses=False  # Important for session storage
)

try:
    redis_conn.ping()
    logger.info("✅ Successfully connected to Redis")
except redis.ConnectionError as e:
    logger.error(f"❌ Failed to connect to Redis: {str(e)}")
    raise

# =============================================
# FLASK CONFIGURATION
# =============================================

app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# Explicit session configuration
app.config['SESSION_COOKIE_NAME'] = SESSION_COOKIE_NAME
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis_conn
app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=24)
app.config['SESSION_PERMANENT'] = True

# Application settings
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['MAX_AVATAR_SIZE'] = 2 * 1024 * 1024
app.config['COHERE_TIMEOUT'] = 30
app.config['GOOGLE_OAUTH_CACHE_TIMEOUT'] = 300
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['CHAT_HISTORY_LIMIT'] = 50

# =============================================
# EXTENSIONS INITIALIZATION
# =============================================

# Initialize Flask-Session FIRST
Session(app)

# Initialize CORS
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://ai-yenugu.netlify.app",
            "http://localhost:3000",
            "https://ai-yenugu.onrender.com"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"],
        "max_age": 600
    }
})

# Initialize rate limiter
try:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri=f"redis://default:6T18RXJsymUjiqNNkEZgVucArLAyi080@redis-14356.c252.ap-southeast-1-1.ec2.redns.redis-cloud.com:14356",
        default_limits=["200 per day", "50 per hour"],
        strategy="fixed-window"
    )
    logger.info("✅ Rate limiter initialized with Redis")
except Exception as e:
    logger.error(f"❌ Error initializing rate limiter: {e}")
    limiter = Limiter(app=app, key_func=get_remote_address)
    logger.info("⚠️  Using in-memory rate limiting")

# =============================================
# HELPER FUNCTIONS
# =============================================

def requires_drive_connection(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_drive_connected():
            return jsonify({"error": "Google Drive not connected"}), 401
        return f(*args, **kwargs)
    return decorated_function

def handle_api_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except HttpError as e:
            logger.error(f"Google API error: {str(e)}")
            return jsonify({"error": "Google Drive operation failed"}), 500
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error: {str(e)}")
            return jsonify({"error": "External service unavailable"}), 503
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "An unexpected error occurred"}), 500
    return decorated_function

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

# =============================================
# SERVICE CLASSES
# =============================================

class DriveManager:
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(HttpError)
    )
    def get_service(credentials):
        creds = Credentials(
            token=credentials['token'],
            refresh_token=credentials['refresh_token'],
            token_uri=credentials['token_uri'],
            client_id=credentials['client_id'],
            client_secret=credentials['client_secret'],
            scopes=credentials['scopes']
        )
        
        if creds.expired and creds.refresh_token:
            creds.refresh(requests.Request())
            session['credentials'] = {
                **session['credentials'],
                'token': creds.token
            }
            session.modified = True
        
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    
    @staticmethod
    def ensure_folder(service, folder_name):
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        folders = service.files().list(q=query, fields="files(id)", pageSize=1).execute().get('files', [])
        return folders[0]['id'] if folders else service.files().create(
            body={'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'},
            fields='id'
        ).execute()['id']

    @staticmethod
    def upload_file(service, folder_id, filename, content, mime_type):
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing_files = service.files().list(q=query, fields="files(id)", pageSize=1).execute().get('files', [])
        
        file_metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type)
        
        if existing_files:
            return service.files().update(
                fileId=existing_files[0]['id'],
                media_body=media,
                fields='id'
            ).execute()['id']
        return service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()['id']

    @staticmethod
    def download_file(service, folder_id, filename):
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        files = service.files().list(q=query, fields="files(id)", pageSize=1).execute().get('files', [])
        if not files:
            raise FileNotFoundError(f"File {filename} not found")
        
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, service.files().get_media(fileId=files[0]['id']))
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()

class AuthManager:
    @staticmethod
    def get_flow():
        client_config = {
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        }
        
        if os.getenv('FLASK_ENV') != 'production':
            os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        
        return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
    
    @staticmethod
    def get_user_info(credentials):
        creds = Credentials(**credentials)
        return build('oauth2', 'v2', credentials=creds).userinfo().get().execute()

# =============================================
# CORE FUNCTIONALITY
# =============================================

def is_drive_connected():
    if 'credentials' not in session:
        return False
    
    try:
        creds = Credentials(**session['credentials'])
        if creds.expired and creds.refresh_token:
            creds.refresh(requests.Request())
            session['credentials']['token'] = creds.token
            session.modified = True
        return True
    except Exception as e:
        logger.warning(f"Invalid credentials: {str(e)}")
        session.pop('credentials', None)
        session.pop('drive_folder_id', None)
        session.modified = True
        return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def generate_cohere_response(message, chat_history=None):
    if not COHERE_API_KEY:
        raise ValueError("Cohere API key not configured")
    
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "message": message,
        "model": "command",
        "temperature": 0.7
    }
    
    if chat_history:
        data["chat_history"] = [
            {"role": "user" if msg["sender"] == "user" else "chatbot", "message": msg["content"]}
            for msg in chat_history if msg["sender"] in ["user", "ai"]
        ]
    
    response = requests.post(COHERE_API_URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json().get("text", "No response from AI")

# =============================================
# API ENDPOINTS
# =============================================

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/api/drive-login', methods=['GET'])
@limiter.limit("10 per minute")
def drive_login():
    flow = AuthManager.get_flow()
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        prompt='consent',
        state=secrets.token_urlsafe(32)
    )
    return jsonify({"auth_url": auth_url})

@app.route('/api/drive-callback', methods=['GET'])
def drive_callback():
    flow = AuthManager.get_flow()
    flow.fetch_token(authorization_response=request.url)
    
    session['credentials'] = {
        'token': flow.credentials.token,
        'refresh_token': flow.credentials.refresh_token,
        'token_uri': flow.credentials.token_uri,
        'client_id': flow.credentials.client_id,
        'client_secret': flow.credentials.client_secret,
        'scopes': flow.credentials.scopes
    }
    
    service = DriveManager.get_service(session['credentials'])
    session['drive_folder_id'] = DriveManager.ensure_folder(service, CHATS_FOLDER_NAME)
    session.modified = True
    
    return f"""
    <script>
        window.opener.postMessage({{type: 'auth-success'}}, '{FRONTEND_URL}');
        window.close();
    </script>
    """

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")
def chat():
    data = request.json
    message = validate_chat_message(data.get('message'))
    
    if 'chat_sessions' not in session:
        session['chat_sessions'] = {}
    
    chat_id = data.get('chat_id', str(uuid.uuid4()))
    if chat_id not in session['chat_sessions']:
        session['chat_sessions'][chat_id] = {
            "title": message[:50] + "..." if len(message) > 50 else message,
            "messages": []
        }
    
    session['chat_sessions'][chat_id]['messages'].append({
        "id": str(uuid.uuid4()),
        "content": message,
        "sender": "user",
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    try:
        ai_response = generate_cohere_response(
            message, 
            session['chat_sessions'][chat_id]['messages']
        )
        
        session['chat_sessions'][chat_id]['messages'].append({
            "id": str(uuid.uuid4()),
            "content": ai_response,
            "sender": "ai",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        if is_drive_connected():
            DriveManager.upload_file(
                DriveManager.get_service(session['credentials']),
                session['drive_folder_id'],
                f"chat_{chat_id}.json",
                json.dumps(session['chat_sessions'][chat_id]).encode('utf-8'),
                'application/json'
            )
        
        return jsonify({
            "response": ai_response,
            "chat_id": chat_id
        })
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =============================================
# APPLICATION ENTRY POINT
# =============================================

if __name__ == '__main__':
    if os.getenv('FLASK_ENV') != 'production':
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        app.debug = True
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)