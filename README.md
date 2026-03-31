# EYES-DEFY-ANEMIA — Hemoglobin Prediction System

## Overview

**EYES-DEFY-ANEMIA** is an advanced machine learning-based diagnostic system designed to predict hemoglobin levels and detect anemia risk from medical eye scan images. This system leverages deep learning (EfficientNet-B0) to provide accurate hemoglobin predictions across different age groups and patient demographics, helping healthcare professionals in early detection and diagnosis of anemia.

The system analyzes ocular imaging data and employs clinically validated hemoglobin ranges to classify patient health status, supporting informed clinical decision-making and patient care optimization.


## Project Structure
```text
anemia_project/
├── dataset/
│   ├── India/                # Images organized by patient folder
│   │   └── India.xlsx        # India dataset labels
│   └── Italy/
│       └── Italy.xlsx        # Italy dataset labels
│
├── notebooks/
│   ├── final_training.ipynb  # Main training source (Data Load -> Training)
│   └── convert_intotflite.ipynb # TFLite format conversion
│
├── src/
│   ├── config.py             # Demographic thresholds & constants
│   ├── data_loader.py        # Dataset processing
│   └── utils.py              # Anemic status, risk, & Severity (0-3) logic
│
├── models/
│   ├── anemia_model.h5       # Trained EfficientNet model
│   ├── anemia_model.tflite   # Android-ready converted model
│   └── hgb_scaler.pkl        # Inverse scaler for hemoglobin g/dL
│
├── mobile_app_integration.md # Technical mobile guide (Java code) and Simple mobile guide (Logic workflow)
├── app.py                    # Streamlit Web UI Dashboard
└── requirements.txt
```

## Dataset Information

The system is trained on comprehensive medical datasets from multiple regions:

### India Dataset
- **Samples**: 95 patient cases
- **Structure**: Organized by patient ID folders
- **Labels**: India.xlsx containing hemoglobin measurements

### Italy Dataset
- **Samples**: 123 patient cases
- **Structure**: Organized by patient ID folders
- **Labels**: Italy.xlsx containing hemoglobin measurements

Total samples provide diverse demographic coverage for robust model generalization.


## 🚀 Getting Started (Web app )

### Prerequisites
- Ensure you have Python 3.9+ installed. 


### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
Start the interactive web application:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501/`

**Live Demo:** https://anemiaanalysis-toobarani.streamlit.app/





## 📱 Mobile / Android Integration
    
### ⚠️ Important Note for Mobile Developers
If you want to integrate this model into an Android or any mobile application, you cannot use the `.h5` file directly. The `.h5` format is designed for server-side or desktop use. Mobile apps require the TensorFlow Lite (`.tflite`) format.
    
### ✅ Use the Pre-converted TFLite Model
This repository already includes the mathematically verified converted model:

```bash
models/anemia_model.tflite
```

This file was generated using the `convert_intotflite.ipynb` notebook. You can use it directly in your Android app without any additional conversion steps.
   
### 📲 Android Integration Guides

For a complete step-by-step guide on integrating the `anemia_model.tflite` model into an Android application (including predicting Hemoglobin, checking Anemia Status, assigning Severity 0-3, and calculating Risk %), please refer to the following files:

- 💻 **[Technical Java Guide (mobile_app_integration.md)](mobile_app_integration.md)**: Includes the exact Java boilerplate, matrix loading, and math functions required for deployment.
- 🧠 **[Simple Logic Guide (mobile_app_integration.md)](mobile_app_integration.md)**: A non-technical, simple workflow explainer detailing how model predictions turn into clinical grades.



## Usage

1. **Upload an Image**: Select a medical eye scan image from your device
2. **Input Patient Details**: Enter patient's age group for accurate classification
3. **Get Prediction**: The model predicts hemoglobin level with confidence score
4. **View Results**: Receive dynamic classification (Grade 0: Normal, Grade 1: Mild, Grade 2: Moderate, Grade 3: Severe) and Risk % probability based on clinical ranges.


## Normal Hemoglobin Ranges (g/dL)
Diagnosis is based on the following clinically validated reference ranges. Values below the minimum indicate anemia, while values exceeding the maximum are flagged as High/Severe hemoglobin levels.

| Category | Age Bracket | Normal Range |
| :--- | :--- | :--- |
| Children | 6 – 59 months | ≥ 11.0 g/dL |
| Children | 5 – 11 yrs | ≥ 11.5 g/dL |
| Children | 12 – 14 yrs | ≥ 12.0 g/dL |
| Non-pregnant Women | 15+ yrs | ≥ 12.0 g/dL |
| Pregnant Women | All ages | ≥ 11.0 g/dL |
| Men | 15+ yrs | ≥ 13.0 g/dL |

## Technologies & Libraries

- **Deep Learning Framework**: TensorFlow / Keras
- **Model Architecture**: EfficientNet-B0 (Pre-trained ImageNet weights)
- **Data Processing**: NumPy, Pandas, OpenCV
- **Image Scaling**: Scikit-learn (MinMaxScaler)
- **Web Interface**: Streamlit
- **Data Management**: Excel (XLSX)
- **Python Version**: 3.8+


## Model Architecture

- **Base Model**: EfficientNet-B0
- **Input Size**: 224 x 224 x 3 (RGB images)
- **Transfer Learning**: Pre-trained on ImageNet weights
- **Output**: Continuous hemoglobin value prediction
- **Activation**: ReLU layers with Batch Normalization
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error (MSE)



### Common Issues

**Issue**: Module not found errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Port 8501 already in use
- **Solution**: Use custom port: `streamlit run app.py --server.port 8502`

**Issue**: Image upload errors
- **Solution**: Ensure images are in supported formats (JPG, PNG) with reasonable file size




**Last Updated**: March 2026  
**Version**: 1.0.0  
**Status**: Active & Maintained
**Develop by** : Tooba Rani
