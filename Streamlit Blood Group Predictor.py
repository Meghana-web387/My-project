# app_with_prediction.py
# This is a complete Streamlit application that communicates with the Mantra MFS110 RD Service,
# captures a fingerprint, and uses a PyTorch model to predict blood group.
#
# To run this application:
# 1. Ensure you have the Mantra RD Service installed and running.
# 2. Make sure your fingerprint device is connected.
# 3. Install the required Python libraries:
#    pip install streamlit requests torch torchvision Pillow
# 4. Save this code as `app_with_prediction.py`.
# 5. Run the command: streamlit run app_with_prediction.py

import streamlit as st
import requests
import xml.etree.ElementTree as ET
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --- Helper Functions for RD Service Communication ---

# List of common ports for the Mantra RD Service.
RD_SERVICE_PORTS = [11100, 11101, 11102]
TIMEOUT_SECONDS = 20

def find_rd_service_url():
    """Tries to find the active RD Service port and returns the base URL."""
    for port in RD_SERVICE_PORTS:
        url = f"http://localhost:{port}"
        try:
            # The RDSERVICE method is a custom method used by Mantra's service
            response = requests.request("RDSERVICE", url, timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                st.success(f"RD Service found on port {port}!")
                return url
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to RD Service on port {port}: {e}")
    st.error("Error: Could not find RD Service. Please ensure it is running and your device is connected.")
    return None

def capture_fingerprint(base_url):
    """Sends a capture request to the RD Service and returns the XML response."""
    pid_options_xml = """<PidOptions ver="1.0">
        <Opts fCount="1" fType="0" iCount="0" pCount="0" format="0" pidVer="2.0" timeout="10000" posh="UNKNOWN" env="P" wadh=""/>
        <Demo/>
        <CustOpts/>
    </PidOptions>"""
    
    try:
        headers = {'Content-Type': 'text/xml'}
        response = requests.request("CAPTURE", f"{base_url}/rd/capture", data=pid_options_xml.encode('utf-8'), headers=headers, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error capturing fingerprint: {e}")
        return None

# --- Blood Group Prediction Logic using PyTorch ---

# Mapping from class index to blood group
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

@st.cache_resource
def load_model(model_path):
    """
    Loads the ResNet-18 model and the pre-trained weights.
    The function is memoized with st.cache_resource to load the model only once.
    """
    try:
        # Load the pre-trained ResNet-18 model
        model = models.resnet18(weights=None)
        
        # Modify the final fully connected layer for our 8 classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(BLOOD_GROUPS))
        
        # Load the saved state dictionary from the uploaded file
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Set the model to evaluation mode
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_fingerprint(base64_data):
    """
    Decodes the base64 fingerprint data and preprocesses it for the model.
    Assumes the base64 data is a standard image format (like PNG or JPEG).
    """
    try:
        # Decode the base64 string to bytes
        img_bytes = base64.b64decode(base64_data)
        
        # Open the image from the byte stream
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Define the transformations required by the model
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Apply transformations and add a batch dimension
        img_tensor = preprocess(img).unsqueeze(0)
        return img_tensor
        
    except Exception as e:
        st.error(f"Error preprocessing fingerprint image: {e}")
        return None

def predict_blood_group(model, fingerprint_data):
    """
    Uses the loaded PyTorch model to predict the blood group.
    """
    try:
        # Preprocess the fingerprint data
        input_tensor = preprocess_fingerprint(fingerprint_data)
        if input_tensor is None:
            return None
        
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get the predicted class index
        _, predicted_idx = torch.max(output, 1)
        
        # Map the index to the blood group string
        predicted_group = BLOOD_GROUPS[predicted_idx.item()]
        
        return predicted_group

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# --- Streamlit UI and Logic ---

st.title("Mantra MFS110 Blood Group Predictor")
st.markdown("Use this app to capture a fingerprint and get a blood group prediction.")

# File uploader for the model weights
uploaded_model_file = st.file_uploader(
    "Upload your `fingerprint_blood_group_resnet18.pth` model file",
    type="pth"
)

if uploaded_model_file is not None:
    # Load the model from the uploaded file
    model = load_model(uploaded_model_file)
    if model:
        st.success("Model is ready!")

        # Find the RD Service when the app starts.
        @st.cache_data
        def get_service_url_cached():
            return find_rd_service_url()
        
        base_url = get_service_url_cached()
        
        if base_url:
            if st.button("Capture Fingerprint & Predict Blood Group"):
                with st.spinner("Waiting for fingerprint capture..."):
                    response_xml = capture_fingerprint(base_url)
        
                if response_xml:
                    try:
                        # Parse the XML response
                        root = ET.fromstring(response_xml)
                        resp_tag = root.find("Resp")
                        err_code = resp_tag.get("errCode") if resp_tag is not None else "-1"
                        err_info = resp_tag.get("errInfo") if resp_tag is not None else "Unknown Error"
        
                        if err_code == "0":
                            st.success("Fingerprint captured successfully! Now predicting...")
                            
                            # Find the Sg element which contains the base64-encoded fingerprint data
                            sg_tag = root.find(".//Sg")
                            
                            if sg_tag is not None:
                                fingerprint_data = sg_tag.text
                                
                                # Use the actual prediction function
                                predicted_group = predict_blood_group(model, fingerprint_data)
                                
                                if predicted_group:
                                    st.balloons()
                                    st.markdown(f"## Predicted Blood Group: **{predicted_group}**")
                                    st.success("Prediction complete!")
                                
                            else:
                                st.error("Error: Could not find fingerprint data (Sg tag) in the response.")
                                
                        else:
                            st.error(f"Capture failed: {err_info}")
                    
                    except ET.ParseError:
                        st.error("Failed to parse XML response from the device.")
        
        else:
            st.error("RD Service not found. Please ensure it is running and your device is connected.")

else:
    st.info("Please upload your `.pth` model file to begin.")
