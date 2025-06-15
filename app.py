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
from urllib.parse import quote_plus, urlparse, parse_qs
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, session, Response, send_file, redirect
from flask_cors import CORS
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from limits.storage import MemoryStorage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis
from redis.exceptions import RedisError, ConnectionError
import msgpack
import jwt
from jwt.exceptions import InvalidTokenError

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
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://your-app.onrender.com/api/drive-callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PROFILE_FILENAME = "user_profile.json"
AVATAR_FILENAME_PREFIX = "user_avatar"
CHATS_FOLDER_NAME = "AI Chat Storage"
COHERE_API_URL = "https://api.cohere.ai/v1/chat"
DEFAULT_CHAT_TITLE = "New Chat"
MAX_CHAT_MESSAGE_LENGTH = 15000  # Increased for complex problems
MAX_CHAT_TITLE_LENGTH = 100
SESSION_COOKIE_NAME = 'ai_session'
CACHE_TTL = 3600
THREAD_POOL_SIZE = 8
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 30

# Initialize thread pool
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

# Redis Configuration
def get_redis_connection():
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        redis_conn = redis.Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_timeout=2,
            socket_connect_timeout=2,
            retry_on_timeout=True,
            max_connections=20,
            health_check_interval=30
        )
        redis_conn.ping()
        logger.info("✅ Successfully connected to Redis")
        return redis_conn
    except (RedisError, ConnectionError) as e:
        logger.error(f"❌ Failed to connect to Redis: {str(e)}")
        logger.warning("Using in-memory storage as fallback")
        return None

redis_conn = get_redis_connection()

# Application Configuration
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config.update({
    'SECRET_KEY': os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32)),
    'SESSION_COOKIE_NAME': SESSION_COOKIE_NAME,
    'SESSION_COOKIE_SECURE': os.getenv('FLASK_ENV') == 'production',
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'SESSION_REFRESH_EACH_REQUEST': True,
    'PERMANENT_SESSION_LIFETIME': datetime.timedelta(minutes=JWT_EXPIRE_MINUTES),
    'SESSION_TYPE': 'redis' if redis_conn else 'filesystem',
    'SESSION_REDIS': redis_conn,
    'SESSION_PERMANENT': True,
    'SESSION_USE_SIGNER': True,
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'webp'},
    'MAX_AVATAR_SIZE': 2 * 1024 * 1024,
    'COHERE_TIMEOUT': 30,  # Increased timeout for complex responses
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'CHAT_HISTORY_LIMIT': 50,
    'JSONIFY_PRETTYPRINT_REGULAR': False,
    'JSON_SORT_KEYS': False,
})

# Initialize Flask-Session
Session(app)

# Initialize CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", FRONTEND_URL],
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "expose_headers": ["Content-Type", "X-CSRFToken"],
        "max_age": 600
    }
})

# JWT Utilities
def create_jwt_token(user_id, email):
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXPIRE_MINUTES),
        "iat": datetime.datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except InvalidTokenError:
        return None

# Rate Limiter
try:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri=os.getenv('REDIS_URL') if redis_conn else "memory://",
        storage_options={
            "socket_timeout": 2,
            "socket_connect_timeout": 2,
            "retry_on_timeout": True,
            "health_check_interval": 30
        } if redis_conn else {},
        strategy="fixed-window",
        default_limits=["500 per day", "100 per hour"],
        headers_enabled=True,
        on_breach=lambda _: None,
        key_prefix="fast_limiter"
    )
    logger.info("✅ Rate limiter initialized")
except Exception as e:
    logger.error(f"❌ Error initializing rate limiter: {e}")
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri="memory://",
        strategy="fixed-window",
        default_limits=["500 per day", "100 per hour"]
    )

@app.errorhandler(RateLimitExceeded)
def handle_rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

@app.errorhandler(RedisError)
def handle_redis_error(e):
    logger.error(f"Redis error: {str(e)}")
    return jsonify({
        "error": "System temporarily unavailable",
        "status": "service_unavailable"
    }), 503

@app.after_request
def after_request(response):
    if request.method == 'OPTIONS':
        return response

    origin = request.headers.get('Origin', '')
    allowed_origins = ['http://localhost:3000', FRONTEND_URL]
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    else:
        response.headers['Cache-Control'] = 'public, max-age=3600'
    
    return response

# Utility functions
def cache_response(ttl=CACHE_TTL, key_prefix='fast_cache'):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if app.config.get('TESTING', False):
                return f(*args, **kwargs)
                
            cache_key = f"{key_prefix}:{request.path}:{hash(frozenset(request.args.items()))}"
            
            try:
                if redis_conn:
                    cached = redis_conn.get(cache_key)
                    if cached:
                        return Response(
                            cached,
                            content_type='application/json',
                            status=200
                        )
            except RedisError:
                pass
                
            result = f(*args, **kwargs)
            
            try:
                if redis_conn and result.status_code == 200:
                    redis_conn.setex(
                        cache_key,
                        ttl,
                        result.get_data()
                    )
            except RedisError:
                pass
                
            return result
        return wrapped
    return decorator

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

# Drive Manager with caching
class DriveManager:
    _service_cache = {}
    _folder_cache = {}
    
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

        return Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(HttpError)
    )
    def get_service(credentials):
        cache_key = f"service:{credentials['token']}"
        if cache_key in DriveManager._service_cache:
            return DriveManager._service_cache[cache_key]
            
        creds = Credentials(
            token=credentials['token'],
            refresh_token=credentials['refresh_token'],
            token_uri=credentials['token_uri'],
            client_id=credentials['client_id'],
            client_secret=credentials['client_secret'],
            scopes=credentials['scopes']
        )

        if creds.expired and creds.refresh_token:
            def refresh_token():
                try:
                    creds.refresh(requests.Request())
                    session['credentials']['token'] = creds.token
                    session.modified = True
                except Exception as e:
                    logger.error(f"Token refresh failed: {str(e)}")

            executor.submit(refresh_token)

        service = build('drive', 'v3', credentials=creds, cache_discovery=False, static_discovery=False)
        DriveManager._service_cache[cache_key] = service
        return service

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(HttpError))
    def ensure_folder(service, folder_name):
        cache_key = f"folder:{folder_name}"
        if cache_key in DriveManager._folder_cache:
            return DriveManager._folder_cache[cache_key]
            
        try:
            if redis_conn:
                cached = redis_conn.get(cache_key)
                if cached:
                    DriveManager._folder_cache[cache_key] = cached.decode('utf-8')
                    return cached.decode('utf-8')
        except RedisError:
            pass
            
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        folders = service.files().list(
            q=query,
            fields="files(id)",
            pageSize=1
        ).execute().get('files', [])

        if folders:
            folder_id = folders[0]['id']
            DriveManager._folder_cache[cache_key] = folder_id
            try:
                if redis_conn:
                    redis_conn.setex(cache_key, CACHE_TTL, folder_id)
            except RedisError:
                pass
            return folder_id

        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        
        DriveManager._folder_cache[cache_key] = folder['id']
        try:
            if redis_conn:
                redis_conn.setex(cache_key, CACHE_TTL, folder['id'])
        except RedisError:
            pass
            
        return folder['id']

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(HttpError))
    def upload_file(service, folder_id, filename, content, mime_type):
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing_files = service.files().list(
            q=query,
            fields="files(id)",
            pageSize=1
        ).execute().get('files', [])

        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type)

        if existing_files:
            file = service.files().update(
                fileId=existing_files[0]['id'],
                media_body=media,
                fields='id'
            ).execute()
        else:
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

        return file['id']

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(HttpError))
    def download_file(service, folder_id, filename):
        cache_key = f"file:{folder_id}:{filename}"
        try:
            if redis_conn:
                cached = redis_conn.get(cache_key)
                if cached:
                    return cached
        except RedisError:
            pass
            
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        files = service.files().list(
            q=query,
            fields="files(id)",
            pageSize=1
        ).execute().get('files', [])

        if not files:
            raise FileNotFoundError(f"File {filename} not found")

        request = service.files().get_media(fileId=files[0]['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        content = fh.getvalue()
        
        try:
            if redis_conn:
                redis_conn.setex(cache_key, CACHE_TTL, content)
        except RedisError:
            pass
            
        return content

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(HttpError))
    def find_avatar_file(service, folder_id):
        query = f"name contains '{AVATAR_FILENAME_PREFIX}' and '{folder_id}' in parents and trashed=false"
        files = service.files().list(
            q=query,
            fields="files(id,name,mimeType,createdTime)",
            orderBy="createdTime desc",
            pageSize=1
        ).execute().get('files', [])

        return files[0] if files else None

    @staticmethod
    def delete_file(service, folder_id, filename):
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        files = service.files().list(
            q=query,
            fields="files(id)",
            pageSize=1
        ).execute().get('files', [])

        if not files:
            return False

        service.files().delete(fileId=files[0]['id']).execute()
        
        cache_key = f"file:{folder_id}:{filename}"
        try:
            if redis_conn:
                redis_conn.delete(cache_key)
        except RedisError:
            pass
            
        return True

    @staticmethod
    def delete_all_avatars(service, folder_id):
        query = f"name contains '{AVATAR_FILENAME_PREFIX}' and '{folder_id}' in parents and trashed=false"
        files = service.files().list(
            q=query,
            fields="files(id)"
        ).execute().get('files', [])

        if not files:
            return False

        for file in files:
            service.files().delete(fileId=file['id']).execute()
            
            cache_key = f"file:{folder_id}:{file['id']}"
            try:
                if redis_conn:
                    redis_conn.delete(cache_key)
            except RedisError:
                pass

        return True

# Auth Manager with JWT
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

        return Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )

    @staticmethod
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type(HttpError))
    def get_user_info(credentials):
        creds = Credentials(
            token=credentials['token'],
            refresh_token=credentials['refresh_token'],
            token_uri=credentials['token_uri'],
            client_id=credentials['client_id'],
            client_secret=credentials['client_secret'],
            scopes=credentials['scopes']
        )

        oauth2_client = build('oauth2', 'v2', credentials=creds, static_discovery=False)
        return oauth2_client.userinfo().get().execute()

    @staticmethod
    def auth_error_response(error_type):
        return f"""
        <html><body><script>
            window.opener.postMessage({{type: 'auth-error', error: '{error_type}'}}, '{FRONTEND_URL}');
            window.close();
        </script></body></html>
        """

    @staticmethod
    def auth_success_response(user_info):
        token = create_jwt_token(user_info.get('id', user_info.get('email')), user_info.get('email'))
        return f"""
        <html><body><script>
            window.opener.postMessage({{
                type: 'auth-success',
                token: '{token}',
                user: {json.dumps(user_info)}
            }}, '{FRONTEND_URL}');
            window.close();
        </script></body></html>
        """

# Profile Manager
class ProfileManager:
    @staticmethod
    def validate_profile_data(data):
        required_fields = ['name', 'email']
        validated = {}

        for field in required_fields:
            if field not in data or not isinstance(data[field], str):
                raise ValueError(f"Invalid or missing field: {field}")
            validated[field] = data[field].strip()

        validated['occupation'] = data.get('occupation', '').strip() if data.get('occupation') else ''
        validated['bio'] = data.get('bio', '').strip() if data.get('bio') else ''

        return validated

    @staticmethod
    def get_default_profile(drive_email):
        return {
            'name': 'New User',
            'email': drive_email,
            'occupation': '',
            'bio': '',
            'avatar_url': '',
            'last_updated': datetime.datetime.now().isoformat()
        }

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    @staticmethod
    def generate_avatar_filename(extension):
        return f"{AVATAR_FILENAME_PREFIX}_{uuid.uuid4().hex[:8]}.{extension}"

# Chat Manager
class ChatManager:
    @staticmethod
    def initialize_chat_session():
        if 'chat_sessions' not in session:
            session['chat_sessions'] = {}
            session.modified = True

        if len(session['chat_sessions']) > app.config['CHAT_HISTORY_LIMIT']:
            sorted_chats = sorted(
                session['chat_sessions'].items(),
                key=lambda x: x[1].get('created_at', '0')
            )
            for chat_id, _ in sorted_chats[:-app.config['CHAT_HISTORY_LIMIT']]:
                del session['chat_sessions'][chat_id]
            session.modified = True

    @staticmethod
    def create_chat_message(content, sender="user", error=False):
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "sender": sender,
            "timestamp": datetime.datetime.now().isoformat(),
            "error": error
        }

    @staticmethod
    def save_chat_to_drive(service, folder_id, chat_id, chat_data):
        def _save():
            try:
                DriveManager.upload_file(
                    service, folder_id,
                    f"chat_{chat_id}.json",
                    msgpack.packb(chat_data),
                    'application/octet-stream'
                )
            except Exception as e:
                logger.error(f"Background save to Drive failed: {str(e)}")

        executor.submit(_save)

    @staticmethod
    def generate_chat_title(messages):
        if not messages:
            return DEFAULT_CHAT_TITLE

        for msg in messages:
            if msg.get('sender') == 'user':
                content = msg.get('content', '')
                if len(content) > MAX_CHAT_TITLE_LENGTH:
                    return content[:MAX_CHAT_TITLE_LENGTH-3] + "..."
                return content

        return DEFAULT_CHAT_TITLE

def is_drive_connected():
    if 'credentials' not in session:
        return False

    try:
        creds = Credentials(**session['credentials'])
        if creds.expired and creds.refresh_token:
            # Async refresh
            def refresh():
                try:
                    creds.refresh(requests.Request())
                    session['credentials']['token'] = creds.token
                    session.modified = True
                except Exception as e:
                    logger.error(f"Token refresh failed: {str(e)}")
            
            executor.submit(refresh)
        return True
    except Exception as e:
        logger.warning(f"Invalid credentials: {str(e)}")
        session.pop('credentials', None)
        session.pop('drive_folder_id', None)
        session.modified = True
        return False

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
        "max_tokens": 2000,  # Increased for complex responses
        "stream": False
    }

    if chat_history:
        data["chat_history"] = [
            {"role": "user" if msg["sender"] == "user" else "chatbot", "message": msg["content"]}
            for msg in chat_history
            if msg["sender"] in ["user", "ai"]
        ]

    try:
        start_time = time.time()
        response = requests.post(
            COHERE_API_URL,
            headers=headers,
            json=data,
            timeout=(10, 30)  # Increased timeouts for complex problems
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
    redis_status = False
    try:
        redis_status = redis_conn.ping() if redis_conn else False
    except RedisError:
        pass

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "cohere": bool(COHERE_API_KEY),
            "google_drive": bool(os.getenv("GOOGLE_CLIENT_ID")),
            "session": True,
            "redis": redis_status
        }
    })

@app.route('/api/drive-auth-url', methods=['GET'])
@limiter.limit("20 per minute")
@handle_api_errors
def drive_auth_url():
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    session['oauth_state_timestamp'] = time.time()
    session.modified = True

    flow = AuthManager.get_flow()
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
        state=state
    )
    return jsonify({"auth_url": auth_url})

@app.route('/api/drive-callback', methods=['GET'])
@handle_api_errors
def drive_callback():
    logger.info(f"OAuth callback received with args: {dict(request.args)}")
    
    state = request.args.get('state')
    stored_state = session.get('oauth_state')
    state_timestamp = session.get('oauth_state_timestamp', 0)

    if not state or not stored_state or not secrets.compare_digest(state, stored_state):
        logger.error(f"State validation failed. Received: {state}, Expected: {stored_state}")
        return AuthManager.auth_error_response("invalid_state")

    if time.time() - state_timestamp > 180:
        logger.error(f"State token expired. Age: {time.time() - state_timestamp} seconds")
        return AuthManager.auth_error_response("expired_state")

    session.pop('oauth_state', None)
    session.pop('oauth_state_timestamp', None)

    flow = AuthManager.get_flow()
    try:
        flow.fetch_token(authorization_response=request.url)
    except Exception as e:
        logger.error(f"Error fetching token: {str(e)}")
        return AuthManager.auth_error_response("token_fetch_failed")

    credentials = flow.credentials
    session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

    try:
        # Parallelize service initialization and user info fetch
        def get_service_and_user():
            service = DriveManager.get_service(session['credentials'])
            session['drive_folder_id'] = DriveManager.ensure_folder(service, CHATS_FOLDER_NAME)
            return AuthManager.get_user_info(session['credentials'])

        user_info = get_service_and_user()
        
        session['user_info'] = {
            'id': user_info.get('id', user_info.get('email', '')),
            'email': user_info.get('email', ''),
            'name': user_info.get('name', ''),
            'picture': user_info.get('picture', '')
        }
        session.modified = True
        
        logger.info(f"User authenticated: {user_info.get('email')}")
        return AuthManager.auth_success_response(session['user_info'])

    except Exception as e:
        logger.error(f"Drive initialization failed: {str(e)}")
        session.clear()
        return AuthManager.auth_error_response("drive_init_failed")

@app.route('/api/auth-status', methods=['GET'])
@handle_api_errors
@cache_response(ttl=30)
def auth_status():
    if not is_drive_connected():
        return jsonify({
            "authenticated": False,
            "drive_connected": False,
            "user": None
        })

    user_info = session.get('user_info', {})
    if not user_info:
        try:
            user_info = AuthManager.get_user_info(session['credentials'])
            session['user_info'] = user_info
            session.modified = True
        except Exception as e:
            logger.error(f"Error getting user info: {str(e)}")
            return jsonify({
                "authenticated": False,
                "drive_connected": False,
                "user": None
            })

    return jsonify({
        "authenticated": True,
        "drive_connected": True,
        "user": {
            "email": user_info.get('email', ''),
            "name": user_info.get('name', ''),
            "picture": user_info.get('picture', '')
        }
    })

@app.route('/api/profile', methods=['GET', 'POST'])
@requires_drive_connection
@handle_api_errors
def profile_handler():
    service = DriveManager.get_service(session['credentials'])
    folder_id = session['drive_folder_id']

    if request.method == 'GET':
        try:
            profile_data = msgpack.unpackb(
                DriveManager.download_file(service, folder_id, PROFILE_FILENAME),
                raw=False
            )

            if DriveManager.find_avatar_file(service, folder_id):
                profile_data['avatar_url'] = f"/api/avatar?t={uuid.uuid4().hex[:8]}"

            return jsonify({"profile": profile_data})
        except FileNotFoundError:
            drive_email = session.get('user_info', {}).get('email', 'unknown@example.com')
            return jsonify({
                "profile": ProfileManager.get_default_profile(drive_email)
            })

    elif request.method == 'POST':
        profile_data = request.form.to_dict()

        if 'profile' in request.form:
            try:
                profile_data.update(json.loads(request.form['profile']))
            except json.JSONDecodeError:
                pass

        profile_data = ProfileManager.validate_profile_data(profile_data)
        profile_data['last_updated'] = datetime.datetime.now().isoformat()

        if 'avatar' in request.files:
            avatar_file = request.files['avatar']
            if avatar_file.filename != '' and ProfileManager.allowed_file(avatar_file.filename):
                if avatar_file.content_length > app.config['MAX_AVATAR_SIZE']:
                    raise ValueError("Avatar file too large (max 2MB allowed)")

                avatar_content = avatar_file.read()
                file_ext = secure_filename(avatar_file.filename).split('.')[-1].lower()
                avatar_filename = ProfileManager.generate_avatar_filename(file_ext)

                DriveManager.delete_all_avatars(service, folder_id)

                DriveManager.upload_file(
                    service, folder_id, avatar_filename,
                    avatar_content, avatar_file.mimetype
                )
                profile_data['avatar_url'] = f"/api/avatar?t={uuid.uuid4().hex[:8]}"

        DriveManager.upload_file(
            service, folder_id, PROFILE_FILENAME,
            msgpack.packb(profile_data),
            'application/octet-stream'
        )

        return jsonify({
            "success": True,
            "profile": profile_data,
            "avatar_updated": 'avatar' in request.files
        })

@app.route('/api/avatar', methods=['GET'])
@requires_drive_connection
@handle_api_errors
@cache_response(ttl=86400)
def get_avatar():
    service = DriveManager.get_service(session['credentials'])
    folder_id = session['drive_folder_id']

    avatar_file = DriveManager.find_avatar_file(service, folder_id)
    if not avatar_file:
        return jsonify({"error": "Avatar not found"}), 404

    avatar_data = DriveManager.download_file(service, folder_id, avatar_file['name'])

    mime_type = avatar_file.get('mimeType', 'image/jpeg')
    if not mime_type or mime_type == 'application/octet-stream':
        mime_type = mimetypes.guess_type(avatar_file['name'])[0] or 'image/jpeg'

    response = send_file(
        io.BytesIO(avatar_data),
        mimetype=mime_type,
        as_attachment=False
    )

    response.headers['Cache-Control'] = 'public, max-age=86400'
    response.headers['ETag'] = str(uuid.uuid4())

    return response

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")
@handle_api_errors
def chat():
    data = request.json
    try:
        message = validate_chat_message(data.get('message'))
        chat_id = data.get('chat_id')
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    ChatManager.initialize_chat_session()

    if not chat_id or chat_id not in session['chat_sessions']:
        chat_id = str(uuid.uuid4())
        session['chat_sessions'][chat_id] = {
            "title": message[:MAX_CHAT_TITLE_LENGTH] + "..." if len(message) > MAX_CHAT_TITLE_LENGTH else message,
            "created_at": datetime.datetime.now().isoformat(),
            "messages": []
        }

    user_message = ChatManager.create_chat_message(message, "user")
    session['chat_sessions'][chat_id]["messages"].append(user_message)
    session.modified = True

    try:
        chat_history = session['chat_sessions'][chat_id]["messages"][-10:]

        def generate_response():
            try:
                return generate_cohere_response(message, chat_history)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return None

        future = executor.submit(generate_response)
        ai_response = future.result(timeout=30)  # Increased timeout for complex problems

        if not ai_response:
            raise Exception("Failed to generate response")

        ai_message = ChatManager.create_chat_message(ai_response, "ai")
        session['chat_sessions'][chat_id]["messages"].append(ai_message)

        if session['chat_sessions'][chat_id]["title"] == DEFAULT_CHAT_TITLE:
            session['chat_sessions'][chat_id]["title"] = ChatManager.generate_chat_title(
                session['chat_sessions'][chat_id]["messages"]
            )

        session.modified = True

        if is_drive_connected():
            ChatManager.save_chat_to_drive(
                DriveManager.get_service(session['credentials']),
                session['drive_folder_id'],
                chat_id,
                session['chat_sessions'][chat_id]
            )

        return jsonify({
            "response": ai_response,
            "chat_id": chat_id,
            "message": ai_message
        })

    except Exception as e:
        error_message = ChatManager.create_chat_message(
            f"Sorry, I couldn't process your request. Please try again later.",
            "system",
            True
        )
        session['chat_sessions'][chat_id]["messages"].append(error_message)
        session.modified = True

        return jsonify({
            "error": str(e),
            "chat_id": chat_id,
            "message": error_message
        }), 500

@app.route('/api/chats', methods=['GET'])
@handle_api_errors
@cache_response(ttl=60)
def list_chats():
    chats = []
    session.setdefault('chat_sessions', {})
    ChatManager.initialize_chat_session()

    for chat_id, chat_data in session['chat_sessions'].items():
        chats.append({
            "id": chat_id,
            "title": chat_data.get("title", f"Chat {chat_id[:8]}"),
            "created_at": chat_data.get("created_at"),
            "source": "memory",
            "message_count": len(chat_data.get("messages", []))
        })

    if is_drive_connected():
        try:
            credentials = session.get('credentials')
            folder_id = session.get('drive_folder_id')

            if credentials and folder_id:
                service = DriveManager.get_service(credentials)

                query = (
                    f"mimeType='application/json' and "
                    f"name contains 'chat_' and "
                    f"'{folder_id}' in parents and trashed=false"
                )

                files = service.files().list(
                    q=query,
                    fields="files(id,name,createdTime)",
                    orderBy="createdTime desc"
                ).execute().get('files', [])

                for file in files:
                    try:
                        name = file.get('name', '')
                        if name.startswith('chat_') and name.endswith('.json'):
                            file_chat_id = name[5:-5]

                            if not any(c['id'] == file_chat_id for c in chats):
                                chats.append({
                                    "id": file_chat_id,
                                    "title": f"Chat {file_chat_id[:8]}...",
                                    "created_at": file.get('createdTime'),
                                    "source": "drive",
                                    "message_count": 0
                                })
                    except Exception as e:
                        logger.error(f"Error processing file {file.get('name')}: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading chats from Drive: {str(e)}")

    sorted_chats = sorted(
        chats,
        key=lambda x: x.get('created_at') or '0',
        reverse=True
    )

    return jsonify({"chats": sorted_chats})

@app.route('/api/chat/<chat_id>', methods=['GET'])
@handle_api_errors
@cache_response(ttl=30)
def get_chat(chat_id):
    ChatManager.initialize_chat_session()

    if chat_id in session['chat_sessions']:
        chat_data = session['chat_sessions'][chat_id]
        return jsonify({
            "status": "success",
            "chat_id": chat_id,
            "title": chat_data.get("title"),
            "created_at": chat_data.get("created_at"),
            "messages": chat_data.get("messages", []),
            "source": "memory"
        })

    if is_drive_connected():
        try:
            service = DriveManager.get_service(session['credentials'])
            folder_id = session['drive_folder_id']
            chat_data = msgpack.unpackb(
                DriveManager.download_file(service, folder_id, f"chat_{chat_id}.json"),
                raw=False
            )

            session['chat_sessions'][chat_id] = chat_data
            session.modified = True

            return jsonify({
                "status": "success",
                "chat_id": chat_id,
                "title": chat_data.get("title"),
                "created_at": chat_data.get("created_at"),
                "messages": chat_data.get("messages", []),
                "source": "drive"
            })
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error loading chat from Drive: {str(e)}")

    return jsonify({
        "status": "error",
        "message": "Chat not found"
    }), 404

@app.route('/api/chat/<chat_id>', methods=['DELETE'])
@handle_api_errors
def delete_chat(chat_id):
    ChatManager.initialize_chat_session()
    deleted_from = []

    if chat_id in session['chat_sessions']:
        del session['chat_sessions'][chat_id]
        session.modified = True
        deleted_from.append("memory")

    if is_drive_connected():
        service = DriveManager.get_service(session['credentials'])
        folder_id = session['drive_folder_id']
        if DriveManager.delete_file(service, folder_id, f"chat_{chat_id}.json"):
            deleted_from.append("drive")

    if not deleted_from:
        return jsonify({
            "status": "error",
            "message": "Chat not found"
        }), 404

    return jsonify({
        "status": "success",
        "deleted_from": deleted_from
    })

@app.route('/api/chat/<chat_id>/title', methods=['PUT'])
@handle_api_errors
def update_chat_title(chat_id):
    try:
        new_title = validate_chat_title(request.json.get('title'))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    ChatManager.initialize_chat_session()
    updated_in = []

    if chat_id in session['chat_sessions']:
        session['chat_sessions'][chat_id]['title'] = new_title
        session.modified = True
        updated_in.append("memory")

    if is_drive_connected():
        try:
            service = DriveManager.get_service(session['credentials'])
            folder_id = session['drive_folder_id']
            chat_data = msgpack.unpackb(
                DriveManager.download_file(service, folder_id, f"chat_{chat_id}.json"),
                raw=False
            )
            chat_data['title'] = new_title
            DriveManager.upload_file(
                service, folder_id,
                f"chat_{chat_id}.json",
                msgpack.packb(chat_data),
                'application/octet-stream'
            )
            updated_in.append("drive")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Error updating chat title in Drive: {str(e)}")

    if not updated_in:
        return jsonify({
            "status": "error",
            "message": "Chat not found"
        }), 404

    return jsonify({
        "status": "success",
        "updated_in": updated_in,
        "new_title": new_title
    })

@app.route('/api/logout', methods=['POST'])
@handle_api_errors
def logout():
    if 'credentials' in session:
        try:
            creds = Credentials(**session['credentials'])
            requests.post(
                'https://oauth2.googleapis.com/revoke',
                params={'token': creds.token},
                headers={'content-type': 'application/x-www-form-urlencoded'},
                timeout=3
            )
        except Exception as e:
            logger.warning(f"Error revoking token: {str(e)}")

    session.clear()
    return jsonify({"success": True})

if __name__ == '__main__':
    if os.getenv('FLASK_ENV') != 'production':
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        app.debug = True

    logger.info("Starting optimized Flask server")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)