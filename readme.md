# 🩸 Fingerprint Based Blood Group Predictor

An advanced AI-powered application that predicts blood groups from fingerprint images using deep learning, with integrated live biometric device capture.

## 🚀 Features

- **AI-Powered Prediction**: Uses ResNet-18 model for accurate blood group classification
- **Live Biometric Capture**: Direct integration with Mantra MFS110 and compatible devices
- **File Upload Support**: Upload fingerprint images for analysis
- **Real-time Processing**: Instant predictions with confidence scoring
- **Professional Logging**: Comprehensive logging for debugging and monitoring
- **Modern UI**: Clean, responsive Streamlit interface

## 📋 Requirements

### Hardware
- Mantra MFS110 or compatible biometric fingerprint scanner
- Computer with USB ports
- Windows OS (for RD Service compatibility)

### Software
- Python 3.8 or higher
- RD Service (for biometric device communication)
- Modern web browser

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd fingerprint-blood-group-predictor
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup RD Service
1. Install the RD Service software for your biometric device
2. Ensure the service is running on one of these ports: 11100, 11101, or 11102
3. Connect your biometric device via USB

### 4. Download/Place Model File
- Ensure `fingerprint_blood_group_resnet18.pth` is in the project directory
- This should be your trained PyTorch model file

### 5. Create Required Directories
```bash
mkdir static
# Place any training graphs/charts in the static folder
```

## 🚀 Running the Application

### 1. Start the Streamlit App
```bash
streamlit run app.py
```

### 2. Access the Application
- Open your browser and navigate to `http://localhost:8501`
- The application will automatically load

### 3. Using Live Capture
1. Navigate to the "📱 Live Capture" tab
2. Click "📱 Get Device Info" to verify device connection
3. Click "👆 Capture Fingerprint" and place finger on scanner
4. Click "🧠 Predict Blood Group" to get results

### 4. Using File Upload
1. Navigate to the "🔍 Predict" tab
2. Upload a fingerprint image (BMP, PNG, JPG)
3. View the AI analysis and prediction results

## 📁 Project Structure

```
fingerprint-blood-group-predictor/
├── app.py                          # Main Streamlit application
├── model.py                        # Model definition and mappings
├── requirements.txt                 # Python dependencies
├── fingerprint_blood_group_resnet18.pth  # Trained model file
├── static/                         # Static files (graphs, charts)
│   ├── training_loss.png
│   ├── test_accuracy.png
│   ├── confusion_matrix.png
│   └── classification_report.png
├── app.log                         # Application logs
└── README.md                       # This file
```

## 🔧 Configuration

### Logging
- Logs are written to `app.log`
- Log level can be adjusted in the `logging.basicConfig()` call
- Logs include timestamps, levels, and detailed messages

### Model Configuration
- Model expects 224x224 grayscale images
- Supports 8 blood group classes: A+, A-, B+, B-, AB+, AB-, O+, O-
- Confidence threshold can be adjusted in the prediction function

### RD Service Configuration
- Default ports: 11100, 11101, 11102
- Timeout: 20 seconds
- PID Options can be modified in the HTML component

## 🎯 Usage Tips

### For Best Results
- Ensure clean fingerprint scanner surface
- Use proper finger placement techniques
- Upload high-quality, clear fingerprint images
- Check device connectivity before capture

### Troubleshooting
- Check `app.log` for detailed error messages
- Verify RD Service is running and device is connected
- Ensure model file is present and compatible
- Test device connection using "Get Device Info"

## 📊 Model Performance

- **Architecture**: Modified ResNet-18 for grayscale input
- **Classes**: 8 blood group types
- **Input Size**: 224×224 pixels
- **Confidence Scoring**: Softmax probability output

## ⚠️ Disclaimer

This application is for **research and educational purposes only**. It should not be used as a substitute for professional medical blood typing. Always consult healthcare professionals for accurate blood group determination.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📧 Contact

**Developer**: Meghana  
**Institution**: Adichunchanagiri Institute Of Technology  
**Year**: 2025  

For questions or support, please contact through official institutional channels.

## 📜 License

This project is for educational and research purposes. Please respect the terms and conditions of the institution and any applicable licenses.

---

**© 2025 Meghana | Adichunchanagiri Institute Of Technology**