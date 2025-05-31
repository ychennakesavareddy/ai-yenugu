import firebase_admin
from firebase_admin import credentials, firestore, auth

def initialize_firebase():
    """
    Initialize the Firebase Admin SDK using the service account credentials.
    """
    try:
        # Replace "firebase-adminsdk.json" with the path to your Firebase service account JSON file
        cred = credentials.Certificate("firebase-adminsdk.json")  # Update this path
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")

# Initialize Firebase when this module is imported
initialize_firebase()

# Optional: Export Firestore and Auth instances for easy access
db = firestore.client()  # Firestore instance
auth_client = auth  # Auth instance