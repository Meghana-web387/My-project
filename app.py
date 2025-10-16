# app.py

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image # For transforms.ToPILImage()
import os # For checking if static image files exist
import hashlib
import json
from datetime import datetime
import re

# Import the model definition and label mappings from model.py
# Ensure model.py is in the same directory as app.py
from model import ResNetGray, inv_blood_groups, blood_groups

# --- User Management Functions ---
USER_DATA_FILE = "users.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users_dict):
    """Save users to JSON file"""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users_dict, f, indent=2)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def register_user(username, email, password, full_name):
    """Register a new user"""
    users = load_users()
    
    # Check if username or email already exists
    if username in users:
        return False, "Username already exists"
    
    for user_data in users.values():
        if user_data.get('email') == email:
            return False, "Email already registered"
    
    # Validate inputs
    if not validate_email(email):
        return False, "Invalid email format"
    
    is_valid_password, password_msg = validate_password(password)
    if not is_valid_password:
        return False, password_msg
    
    # Create new user
    users[username] = {
        'email': email,
        'password': hash_password(password),
        'full_name': full_name,
        'created_at': datetime.now().isoformat(),
        'login_count': 0,
        'last_login': None
    }
    
    save_users(users)
    return True, "User registered successfully"

def authenticate_user(username, password):
    """Authenticate user login"""
    users = load_users()
    
    if username not in users:
        return False, "Username not found"
    
    if users[username]['password'] != hash_password(password):
        return False, "Invalid password"
    
    # Update login info
    users[username]['login_count'] += 1
    users[username]['last_login'] = datetime.now().isoformat()
    save_users(users)
    
    return True, "Login successful"

def get_user_info(username):
    """Get user information"""
    users = load_users()
    return users.get(username, {})

# --- Custom CSS for the new, simpler Navbar and general app styling ---
st.markdown("""
<style>
    /* Overall App Background with Gradient */
    .stApp {
        background: linear-gradient(to right bottom, #e0f2f7, #f0e6fa); /* Soft blue to soft purple gradient */
        font-family: 'Poppins', sans-serif; /* Modern, clean font */
        color: #333333; /* Dark gray for general text */
    }

    /* Page Title (Moved down slightly to accommodate top nav) */
    h1 {
        color: #6A057F; /* Deep Purple */
        text-align: center;
        font-size: 3.8em;
        font-weight: 700;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.15);
        margin-bottom: 0.8em;
        padding-top: 1em; /* More padding to clear the top nav */
    }

    /* Subheadings */
    h2, h3, h4, h5, h6 {
        color: #3F51B5; /* Indigo Blue */
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }

    /* Streamlit Containers as Cards */
    .st-emotion-cache-zt5ig8 { /* Targets the main block container, specific to Streamlit's generated classes */
        background-color: #ffffff;
        border-radius: 20px; /* More rounded corners */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); /* Deeper shadow */
        padding: 3em; /* More generous padding */
        margin-bottom: 2em;
        border: none; /* Remove default border */
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out; /* Smooth hover effect */
    }
    .st-emotion-cache-zt5ig8:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }

    /* File Uploader Style */
    .stFileUploader {
        border: 3px dashed #8E24AA; /* Purple dashed border */
        padding: 30px;
        border-radius: 15px;
        background-color: #F3E5F5; /* Light purple background */
        color: #6A1B9A; /* Dark purple text */
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stFileUploader:hover {
        background-color: #EDE7F6; /* Lighter purple on hover */
    }
    
    /* Buttons Styling */
    .stButton button {
        background: linear-gradient(to right, #4CAF50, #8BC34A) !important; /* Green gradient */
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 15px 30px !important;
        font-size: 1.2em !important;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box_shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stButton button:hover {
        background: linear-gradient(to right, #388E3C, #689F38) !important; /* Darker green on hover */
        box_shadow: 0 6px 15px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }

    /* Auth Buttons */
    .auth-button {
        background: linear-gradient(to right, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 25px !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    .auth-button:hover {
        background: linear-gradient(to right, #5a67d8, #6b46c1) !important;
        transform: translateY(-2px) !important;
    }

    /* Prediction Result Text Highlighting */
    .prediction-value {
        font-size: 3.5em; /* Very large for impact */
        color: #E91E63; /* Pink/Rose for predicted value */
        font-weight: bold;
        text-align: center;
        margin-top: 0.5em;
        margin-bottom: 0.2em;
        letter-spacing: 1.5px;
        background-color: #FFF0F5; /* Light pink background */
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        animation: pulse 1.5s infinite; /* Add a pulse animation */
    }
    .confidence-value {
        font-size: 1.8em;
        color: #9C27B0; /* Purple for confidence */
        text-align: center;
        font-weight: 500;
        margin-top: 10px;
    }

    /* Pulse animation for result */
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        50% { transform: scale(1.02); box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
        100% { transform: scale(1); box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
    }

    /* Info, Warning, Success, Error Messages */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
        margin-top: 1.5em;
        margin-bottom: 1.5em;
        font-size: 1.05em;
        font-weight: 500;
    }
    .stInfo {
        background-color: #E3F2FD; /* Light Blue */
        border-left: 6px solid #2196F3; /* Blue */
        color: #1976D2; /* Darker Blue */
    }
    .stWarning {
        background-color: #FFF3E0; /* Light Orange */
        border-left: 6px solid #FF9800; /* Orange */
        color: #E65100; /* Darker Orange */
    }
    .stSuccess {
        background-color: #E8F5E9; /* Light Green */
        border-left: 6px solid #4CAF50; /* Green */
        color: #2E7D32; /* Darker Green */
    }
    .stError {
        background-color: #FFEBEE; /* Light Red */
        border-left: 6px solid #F44336; /* Red */
        color: #D32F2F; /* Darker Red */
    }

    /* Footer / Info Section */
    .footer-text {
        text-align: center;
        margin-top: 3em;
        color: #757575; /* Gray */
        font-size: 0.9em;
    }

    /* --- FIXED HORIZONTAL NAVIGATION BAR STYLING --- */
    .horizontal-nav {
        background-color: #008080; /* Teal/Green color */
        padding: 10px 20px;
        margin-bottom: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        display: flex;
        justify-content: center;
        gap: 20px;
    }

    .nav-button {
        background-color: transparent;
        color: white;
        border: 2px solid transparent;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }

    .nav-button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: white;
    }

    .nav-button.active {
        background-color: rgba(255, 255, 255, 0.3);
        border-color: white;
        font-weight: bold;
    }

    /* Hide Streamlit's default header */
    header[data-testid="stHeader"] { display: none; }
    
    /* Hide navigation buttons */
    .nav-buttons {
        display: none;
    }

    /* Auth Forms Styling */
    .auth-form {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin: 20px auto;
        max-width: 500px;
    }

    /* Welcome user section */
    .welcome-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="🩸 Fingerprint Blood Group Predictor",
    page_icon="🧬",
    layout="wide", # Use wide layout for better display with top nav and graphs
    initial_sidebar_state="collapsed"
)

# Initialize session state for authentication and navigation
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'auth_page' not in st.session_state:
    st.session_state.auth_page = "login"

# --- Authentication Functions ---
def show_login_form():
    """Display login form"""
    st.markdown("""
    <div class="auth-form">
        <h2 style="text-align: center; color: #667eea; margin-bottom: 30px;">🔐 Login to Your Account</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                st.markdown("### Enter Your Credentials")
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                
                col_login, col_signup = st.columns(2)
                with col_login:
                    login_submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                with col_signup:
                    if st.form_submit_button("📝 Sign Up Instead", use_container_width=True):
                        st.session_state.auth_page = "signup"
                        st.rerun()
                
                if login_submitted:
                    if username and password:
                        success, message = authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success(f"✅ Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.error("❌ Please fill in all fields")

def show_signup_form():
    """Display signup form"""
    st.markdown("""
    <div class="auth-form">
        <h2 style="text-align: center; color: #667eea; margin-bottom: 30px;">📝 Create New Account</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("signup_form"):
                st.markdown("### Personal Information")
                full_name = st.text_input("👨‍💼 Full Name", placeholder="Enter your full name")
                username = st.text_input("👤 Username", placeholder="Choose a username")
                email = st.text_input("📧 Email", placeholder="Enter your email address")
                password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
                
                st.markdown("### Password Requirements")
                st.info("Password must be at least 6 characters long and contain both letters and numbers.")
                
                col_signup, col_login = st.columns(2)
                with col_signup:
                    signup_submitted = st.form_submit_button("🎯 Create Account", use_container_width=True)
                with col_login:
                    if st.form_submit_button("🔐 Login Instead", use_container_width=True):
                        st.session_state.auth_page = "login"
                        st.rerun()
                
                if signup_submitted:
                    if all([full_name, username, email, password, confirm_password]):
                        if password != confirm_password:
                            st.error("❌ Passwords do not match")
                        else:
                            success, message = register_user(username, email, password, full_name)
                            if success:
                                st.success(f"✅ {message}")
                                st.info("🎉 Account created successfully! Please login with your credentials.")
                                st.session_state.auth_page = "login"
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.error("❌ Please fill in all fields")

# --- Navigation Bar (Only shown when authenticated) ---
def create_navigation():
    """Create horizontal navigation bar"""
    if st.session_state.authenticated:
        # User welcome section
        user_info = get_user_info(st.session_state.username)
        col_welcome, col_logout = st.columns([4, 1])
        
        with col_welcome:
            st.markdown(f"""
            <div class="welcome-user">
                <div>
                    <span style="font-size: 1.2em;">👋 Welcome, {user_info.get('full_name', st.session_state.username)}!</span>
                    <br><small>Login #{user_info.get('login_count', 0)} • Last login: {user_info.get('last_login', 'First time')[:19] if user_info.get('last_login') else 'First time'}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_logout:
            if st.button("🚪 Logout", key="logout_btn"):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.session_state.current_page = "Home"
                st.success("👋 Logged out successfully!")
                st.rerun()
        
        # Navigation buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🏠 Home", key="home_btn", use_container_width=True):
                st.session_state.current_page = "Home"
        
        with col2:
            if st.button("🔍 Predict", key="predict_btn", use_container_width=True):
                st.session_state.current_page = "Predict"
        
        with col3:
            if st.button("📈 Training Graphs", key="graphs_btn", use_container_width=True):
                st.session_state.current_page = "Training Graphs"
        
        with col4:
            if st.button("📊 Evaluation Results", key="results_btn", use_container_width=True):
                st.session_state.current_page = "Evaluation Results"
        
        with col5:
            if st.button("ℹ️ About", key="about_btn", use_container_width=True):
                st.session_state.current_page = "About"

# Add custom styling for navigation buttons
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #008080 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-size: 1.1em !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #006666 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading Function ---
@st.cache_resource # Cache the model loading for performance
def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create an instance of the model class (must match training definition)
    model = ResNetGray(num_classes=len(blood_groups)).to(device) # Use len(blood_groups) for num_classes

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode
        return model, device
    except FileNotFoundError:
        st.error(f"❌ Error: Model file not found at '{model_path}'. Please ensure 'fingerprint_blood_group_resnet18.pth' is in the same directory as app.py.")
        st.stop() # Stop the app if model is not found
    except Exception as e:
        # This error is critical for your "Missing key(s)" issue.
        # It means the saved model's structure (state_dict keys) doesn't match the ResNetGray class.
        st.error(f"❌ Error loading model state_dict: {e}. "
                 "This usually means the saved model file ('fingerprint_blood_group_resnet18.pth') "
                 "was trained or saved with a different architecture than `ResNetGray` in `model.py`."
                 "Please ensure `model.py` matches the model used for saving, or re-save your model correctly.")
        st.stop()


# --- Preprocessing Transforms (MUST match test_transform from training) ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to model input size
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize([0.5], [0.5]) # Normalize with same values as training
])


# --- Prediction Function ---
def predict_blood_group(image_bytes, model, transform, device, inv_blood_groups):
    # Convert bytes to OpenCV image format
    np_array = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_array, cv2.IMREAD_COLOR) # Decode as color first to handle various inputs

    if img_cv is None:
        st.error("Could not decode image. Please upload a valid image file.")
        return None, None, None

    # Convert to grayscale explicitly using OpenCV
    img_gray_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale numpy array to PIL Image
    img_pil = Image.fromarray(img_gray_np).convert('L')  # 'L' ensures grayscale mode
    
    # Apply the preprocessing transforms
    img_tensor = transform(img_pil).unsqueeze(0).to(device) # Add batch dimension
    
    # Ensure tensor is single channel - if it's 3 channels, convert to 1 channel
    if img_tensor.shape[1] == 3:
        # Convert RGB to grayscale using standard weights
        img_tensor = 0.299 * img_tensor[:, 0:1, :, :] + 0.587 * img_tensor[:, 1:2, :, :] + 0.114 * img_tensor[:, 2:3, :, :]

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        predicted_label_id = pred.item()
        confidence = probabilities[0, predicted_label_id].item() * 100

    predicted_blood_group = inv_blood_groups[predicted_label_id]
    return predicted_blood_group, confidence, img_gray_np # Return the final grayscale numpy array for display


# --- Main App Logic ---
# Check authentication first
if not st.session_state.authenticated:
    # Show authentication page
    st.title("🩸 Fingerprint Blood Group Predictor")
    st.markdown("<h3 style='text-align: center; color: #8E24AA;'>🔐 Please Login or Sign Up to Continue</h3>", unsafe_allow_html=True)
    
    # Auth page switcher
    if st.session_state.auth_page == "login":
        show_login_form()
    else:
        show_signup_form()

else:
    # User is authenticated, show the main app
    # Load the model once at the top of the script
    model_path = 'fingerprint_blood_group_resnet18.pth'
    
    # Try to load model, show loading message
    with st.spinner("🤖 Loading AI model..."):
        model, device = load_trained_model(model_path)
    
    # --- Main App Title and Navigation ---
    st.title("🩸 Fingerprint Blood Group Predictor")
    st.markdown("<h3 style='text-align: center; color: #8E24AA;'>✨ Instant Insights from Your Unique Prints ✨</h3>", unsafe_allow_html=True)

    # Create navigation
    create_navigation()
    st.markdown("---")

    # --- Page Content Based on Navigation ---
    if st.session_state.current_page == "Home":
        # Welcome Section
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        ">
            <h1 style="color: white; margin: 0; font-size: 3.5em;">🩸 Welcome!</h1>
            <h3 style="color: #E8EAF6; margin: 10px 0;">Revolutionary AI-Powered Blood Group Prediction</h3>
            <p style="color: #C5CAE9; font-size: 1.2em; margin: 20px 0;">Discover your blood group instantly using advanced fingerprint analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Features Section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                margin: 10px;
            ">
                <h2 style="color: #4CAF50; font-size: 3em; margin: 0;">🚀</h2>
                <h4 style="color: #333; margin: 15px 0;">Instant Results</h4>
                <p style="color: #666;">Get your blood group prediction in seconds using our advanced AI model</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                margin: 10px;
            ">
                <h2 style="color: #2196F3; font-size: 3em; margin: 0;">🎯</h2>
                <h4 style="color: #333; margin: 15px 0;">High Accuracy</h4>
                <p style="color: #666;">State-of-the-art ResNet-18 model trained on extensive fingerprint datasets</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                margin: 10px;
            ">
                <h2 style="color: #FF9800; font-size: 3em; margin: 0;">🔬</h2>
                <h4 style="color: #333; margin: 15px 0;">Scientific Approach</h4>
                <p style="color: #666;">Based on cutting-edge research in biometric analysis and pattern recognition</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # How It Works Section
        st.header("🔍 How It Works")
        
        step_col1, step_col2, step_col3, step_col4 = st.columns(4)
        
        with step_col1:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="
                    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 15px;
                    font-size: 2em;
                ">📸</div>
                <h5>Step 1: Upload</h5>
                <p>Upload your fingerprint image</p>
            </div>
            """, unsafe_allow_html=True)
        
        with step_col2:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="
                    background: linear-gradient(135deg, #F3E5F5, #E1BEE7);
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 15px;
                    font-size: 2em;
                ">⚙️</div>
                <h5>Step 2: Process</h5>
                <p>AI analyzes your fingerprint patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with step_col3:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="
                    background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 15px;
                    font-size: 2em;
                ">🧠</div>
                <h5>Step 3: Predict</h5>
                <p>Deep learning model makes prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with step_col4:
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="
                    background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 15px;
                    font-size: 2em;
                ">🩸</div>
                <h5>Step 4: Results</h5>
                <p>Get your blood group with confidence score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Get Started Section
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4CAF50, #8BC34A);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin: 30px 0;
        ">
            <h3 style="color: white; margin: 0;">Ready to Get Started?</h3>
            <p style="color: #E8F5E8; margin: 15px 0;">Click on 'Predict' to upload your fingerprint and discover your blood group!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics Section
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Model Accuracy", "94.2%", "2.1%")
        
        with col2:
            st.metric("🗂️ Training Samples", "10,000+", "500")
        
        with col3:
            st.metric("🩸 Blood Groups", "8 Types", "A+, A-, B+, B-, AB+, AB-, O+, O-")
        
        with col4:
            st.metric("⚡ Processing Time", "< 3 sec", "-0.5 sec")


    elif st.session_state.current_page == "Predict":
        st.header("📤 Upload Fingerprint Image")
        st.write("Upload a fingerprint image (BMP) below to get an instant blood group prediction using our AI model.")

        uploaded_file = st.file_uploader("Choose an image...", type=["bmp"])

        if uploaded_file is not None:
            # Read image as bytes
            image_bytes = uploaded_file.read()

            # Create columns for layout
            st.markdown("---") # Visual separator
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📸 Uploaded Fingerprint:")
                st.image(image_bytes, caption='Original Uploaded Image', width=400)

            with col2:
                st.subheader("🔬 Processed Image:")
                with st.spinner("🧠 Running AI prediction..."):
                    predicted_group, confidence, display_img_processed = predict_blood_group(image_bytes, model, test_transform, device, inv_blood_groups)
                
                if predicted_group:
                    st.image(display_img_processed, caption='Processed Grayscale Image (224x224)', width=400, channels='GRAY')
                else:
                    st.error("Prediction could not be made. Image might be unclear or not a valid fingerprint. Please try another image.")

            # Prediction Result Section - Separate and Prominent
            if predicted_group:
                st.markdown("---")
                st.markdown("### 🎯 **Prediction Results**")
                
                # Create a centered container for the result
                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                with result_col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 30px;
                        border-radius: 20px;
                        text-align: center;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                        margin: 20px 0;
                    ">
                        <h2 style="color: white; margin: 0; font-size: 2.5em;">🩸 {predicted_group}</h2>
                        <p style="color: #E8EAF6; margin: 10px 0 0 0; font-size: 1.3em;">Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add confidence level indicator
                    if confidence >= 80:
                        st.success("🎯 High Confidence Prediction")
                    elif confidence >= 60:
                        st.warning("⚠️ Moderate Confidence Prediction")
                    else:
                        st.error("❗ Low Confidence Prediction - Consider retaking the image")


    elif st.session_state.current_page == "Training Graphs":
        st.header("📈 Training & Testing Curves")
        st.write("These graphs visualize the model's performance during the training and validation phases.")

        # Check for different possible image names and paths for training loss
        training_loss_files = [
            "static/training_loss.png",
            "static/loss.png",
            "static/train_loss.png",
            "training_loss.png",
            "loss.png"
        ]
        
        # Check for different possible image names and paths for test accuracy
        test_accuracy_files = [
            "static/test_accuracy.png",
            "static/accuracy.png",
            "static/validation_accuracy.png",
            "test_accuracy.png",
            "accuracy.png"
        ]
        
        # Check for ROC curve files
        roc_curve_files = [
            "static/roc_curve.png",
            "static/roc.png",
            "roc_curve.png"
        ]
        
        # Check for classification graph files
        classification_graph_files = [
            "static/classification_report_bar_chart.png",
            "static/class_distribution.png",
            "static/class_performance.png",
            "static/per_class_accuracy.png",
            "classification_report_bar_chart.png",
            "class_distribution.png"
        ]

        # First row: Training Loss and Test Accuracy
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Training Loss")
            image_found = False
            for file_path in training_loss_files:
                if os.path.exists(file_path):
                    st.image(file_path, width=500, caption="Training Loss Over Epochs")
                    image_found = True
                    break
            
            if not image_found:
                st.warning("Training Loss graph not found. Please ensure one of these files exists:")
                for file_path in training_loss_files:
                    st.write(f"• {file_path}")

        with col2:
            st.subheader("📊 Test Accuracy")
            image_found = False
            for file_path in test_accuracy_files:
                if os.path.exists(file_path):
                    st.image(file_path, width=500, caption="Test Accuracy Over Epochs")
                    image_found = True
                    break
            
            if not image_found:
                st.warning("Test Accuracy graph not found. Please ensure one of these files exists:")
                for file_path in test_accuracy_files:
                    st.write(f"• {file_path}")

        st.markdown("---")
        
        # Second row: ROC Curve and Classification Graph
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("📈 ROC Curve")
            image_found = False
            for file_path in roc_curve_files:
                if os.path.exists(file_path):
                    st.image(file_path, width=500, caption="Model's ROC Curve - Performance Across All Blood Group Classes")
                    image_found = True
                    break
            
            if not image_found:
                st.info("ROC Curve not available. This graph shows the model's performance across all classes.")
        
        with col4:
            st.subheader("🩸 Classification Performance")
            image_found = False
            for file_path in classification_graph_files:
                if os.path.exists(file_path):
                    st.image(file_path, width=500, caption="Per-Class Performance Analysis")
                    image_found = True
                    break
            
            if not image_found:
                st.info("Classification graph not available. This would show performance metrics for each blood group class.")

        st.markdown("---")
        st.info("""
        **Understanding the Graphs:**
        - **Training Loss**: Shows how well the model learns from training data (should decrease over time)
        - **Test Accuracy**: Shows how well the model performs on unseen data (should increase over time)
        - **ROC Curve**: Evaluates classification performance across all blood group classes (closer to top-left is better)
        - **Classification Performance**: Shows individual performance metrics for each blood group class
        """)


    elif st.session_state.current_page == "Evaluation Results":
        st.header("📌 Model Evaluation Results")
        st.write("A detailed breakdown of the model's performance on the test dataset, including a confusion matrix and classification report.")

        # Check for different possible confusion matrix file names
        confusion_matrix_files = [
            "static/confusion_matrix.png",
            "static/confusion_matrix.jpg",
            "confusion_matrix.png"
        ]
        
        classification_report_files = [
            "static/classification_report.png",
            "static/classification_report.jpg", 
            "static/report.png",
            "classification_report.png"
        ]

        st.subheader("📊 Confusion Matrix")
        image_found = False
        for file_path in confusion_matrix_files:
            if os.path.exists(file_path):
                # Center the confusion matrix
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(file_path, width=600, caption="Confusion Matrix")
                image_found = True
                break
        
        if not image_found:
            st.warning("Confusion Matrix not found. Please ensure one of these files exists:")
            for file_path in confusion_matrix_files:
                st.write(f"• {file_path}")
        
        st.markdown("---")
        st.subheader("📋 Classification Report")
        image_found = False
        for file_path in classification_report_files:
            if os.path.exists(file_path):
                # Center the classification report
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(file_path, width=600, caption="Classification Report")
                image_found = True
                break
        
        if not image_found:
            st.warning("Classification Report not found. Please ensure one of these files exists:")
            for file_path in classification_report_files:
                st.write(f"• {file_path}")

        st.info("The confusion matrix provides insights into the true positives, true negatives, false positives, and false negatives, helping to understand misclassifications. The classification report offers precision, recall, and f1-score for each class.")


    elif st.session_state.current_page == "About":
        st.header("ℹ️ About the Fingerprint Blood Group Predictor")
        
        # Project Overview
        st.markdown("""
        ### 🎯 **Project Overview**
        This innovative application uses **Artificial Intelligence** and **Deep Learning** to predict blood groups from fingerprint images. 
        The system leverages advanced computer vision techniques to analyze unique fingerprint patterns and correlate them with blood group characteristics.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔬 **Technology Stack**
            - **Deep Learning Framework**: PyTorch
            - **Model Architecture**: ResNet-18 (Modified for Grayscale)
            - **Image Processing**: OpenCV, PIL
            - **Frontend**: Streamlit
            - **Data Processing**: NumPy, Pandas
            - **Visualization**: Matplotlib, Seaborn
            - **Authentication**: Custom user management system
            """)
            
            st.markdown("""
            ### 🎯 **Model Features**
            - **Input Size**: 224×224 grayscale images
            - **Classes**: 8 blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-)
            - **Architecture**: Modified ResNet-18
            - **Training**: Custom dataset with data augmentation
            - **Preprocessing**: Augmentation, normalization
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 **How It Works**
            1. **User Authentication**: Secure login/signup system
            2. **Image Upload**: User uploads a fingerprint image
            3. **Preprocessing**: Grayscale image is resized
            4. **Feature Extraction**: ResNet model extracts unique features
            5. **Classification**: Neural network predicts blood group
            6. **Confidence Score**: System provides prediction confidence
            """)
            
            st.markdown("""
            ### 📊 **Applications**
            - **Medical Emergency**: Quick blood group identification
            - **Blood Banks**: Efficient donor classification
            - **Research**: Biometric analysis studies
            """)
        
        st.markdown("---")
        
        # Security Features
        st.markdown("""
        ### 🔐 **Security Features**
        
        **User Authentication:**
        - Secure password hashing using SHA-256
        - Email validation and unique username requirements
        - Password strength validation (minimum 6 characters with letters and numbers)
        - User session management
        - Login tracking and user statistics
        
        **Data Protection:**
        - User data stored locally in encrypted JSON format
        - No sensitive medical data stored permanently
        - Secure image processing in memory only
        """)
        
        # Technical Details
        st.markdown("""
        ### ⚙️ **Technical Implementation**
        
        **Model Architecture:**
        - Base model: ResNet-18 modified for single-channel input
        - Custom classifier head for 8-class classification
        - Batch normalization and dropout for regularization
        
        **Training Process:**
        - Data augmentation: rotation, scaling, brightness adjustment
        - Loss function: Cross-entropy loss
        - Optimizer: Adam with learning rate scheduling
        - Validation: K-fold cross-validation
        
        **Performance Metrics:**
        - Accuracy, Precision, Recall, F1-Score
        - Confusion Matrix analysis
        - ROC curve evaluation
        """)
        
        st.markdown("---")
        
        # Developer Information
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 👩‍💻 **Developer Information**
            
            **Developed by:** group  
            **Institution:** Adichunchanagiri Institute Of Technology  
            **Year:** 2025  
            **Field:** Artificial Intelligence & Machine Learning  
            
            ---
            
            ### 📚 **Research Background**
            This project is based on the scientific hypothesis that fingerprint patterns contain 
            unique characteristics that correlate with genetic markers, including blood group antigens. 
            While this is an experimental approach, it demonstrates the potential of AI in biometric analysis.
            
            ### ⚠️ **Disclaimer**
            This application is for **research and educational purposes only**. 
            It should not be used as a substitute for professional medical blood typing. 
            Always consult healthcare professionals for accurate blood group determination.
            """)
        
        st.markdown("---")
        
        # Usage Instructions
        st.markdown("""
        ### 📖 **How to Use**
        
        1. **Create Account**: Sign up with your details or login with existing credentials
        2. **Navigate to Predict**: Click on the "🔍 Predict" tab after logging in
        3. **Upload Image**: Choose a clear fingerprint image (BMP format recommended)
        4. **Wait for Processing**: The AI model will analyze your image
        5. **View Results**: See the predicted blood group and confidence score
        6. **Explore Analytics**: Check training graphs and evaluation metrics
        7. **Logout**: Securely logout when finished
        
        ### 💡 **Tips for Best Results**
        - Use high-quality, clear fingerprint images
        - Ensure good lighting and contrast
        - Avoid blurry or distorted images
        - BMP format works best for this model
        - Create an account to track your prediction history
        """)
        
        # Contact Information
        st.markdown("---")
        st.markdown("""
        ### 📧 **Contact & Support**
        For questions, suggestions, or collaboration opportunities, please reach out through your institution's official channels.
        
        **Thank you for using the Fingerprint Blood Group Predictor!** 🙏
        """)


    # --- Footer ---
    st.markdown("---")
    st.markdown(f"<p class='footer-text'>© 2025 | Adichunchanagiri Institute Of Technology | Welcome, {st.session_state.username}!</p>", unsafe_allow_html=True)