# EYES-DEFY-ANEMIA — Hemoglobin Prediction System

## Overview

**EYES-DEFY-ANEMIA** is an advanced machine learning-based diagnostic system designed to predict hemoglobin levels and detect anemia risk from medical eye scan images. This system leverages deep learning (EfficientNet-B0) to provide accurate hemoglobin predictions across different age groups and patient demographics, helping healthcare professionals in early detection and diagnosis of anemia.

The system analyzes ocular imaging data and employs clinically validated hemoglobin ranges to classify patient health status, supporting informed clinical decision-making and patient care optimization.

## Features

- 🔬 **AI-Powered Prediction**: Uses EfficientNet-B0 deep learning model for accurate hemoglobin level estimation
- 🌍 **Multi-Regional Dataset**: Trained on diverse datasets from India and Italy for improved generalization
- 📊 **Clinical Classification**: Automatic categorization based on age-specific hemoglobin ranges
- 🎯 **High Accuracy**: Optimized model achieving robust performance across different demographics
- 🚀 **Easy-to-Use Interface**: Interactive Streamlit web application for quick predictions
- 📈 **Comprehensive Analysis**: Detailed reporting with classification of Anemia, Normal, and High Hemoglobin levels
- ☁️ **Cloud Deployment**: Live application accessible at https://anemiaanalysis-toobarani.streamlit.app/

## Project Structure
```
anemia_project/
├── dataset/
│   ├── India/              # Images organized by patient folder
│   │   ├── ...
│   │   └── India.xlsx      # Label file
│   └── Italy/
│       ├── ...
│       └── Italy.xlsx
│
├── notebooks/
│   └── final_training.ipynb # Main training source (Data Load -> Training)
│
├── src/
│   ├── config.py           # Range configurations & constants
│   ├── data_loader.py      # Dataset processing
│   ├── model.py            # EfficientNet-B0 architecture
│   └── utils.py            # Range-based classification & inference
│
├── models/
│   └── anemia_model.h5     # Trained EfficientNet model
│
├── app.py                  # Streamlit UI
└── requirements.txt
```

## Setup & Run

### Prerequisites
- **Python 3.8+** installed on your system
- **Git** for version control
- **pip** package manager

### 1. Install Dependencies
Clone the repository and install all required packages:
```bash
git clone <repository-url>
cd Anemia_prediction
pip install -r requirements.txt
```

### 2. Model Training (Optional)
To retrain the model with your own data:
1. Prepare your dataset in the `dataset/` directory with proper folder structure
2. Open and execute the [Jupyter Notebook](notebooks/final_training.ipynb)
3. The notebook includes:
   - Data loading from multiple regions
   - Image preprocessing and normalization (MinMaxScaler)
   - EfficientNet-B0 model architecture and training
   - Model evaluation and validation
4. The trained model will be saved to `models/` directory

### 3. Launch Application
Start the interactive web application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501/`

**Live Demo:** https://anemiaanalysis-toobarani.streamlit.app/

## Usage

1. **Upload an Image**: Select a medical eye scan image from your device
2. **Input Patient Details**: Enter patient's age group for accurate classification
3. **Get Prediction**: The model predicts hemoglobin level with confidence score
4. **View Results**: Receive classification (Anemia/Normal/High) based on clinical ranges
5. **Download Report**: Export the analysis results as needed

## Normal Hemoglobin Ranges (g/dL)
Diagnosis is based on the following clinically validated reference ranges. Values below the minimum indicate anemia, while values exceeding the maximum are flagged as High/Severe hemoglobin levels.

| Category | Age Bracket | Normal Range |
| :--- | :--- | :--- |
| Children | 0 – 10 yrs | 11.0 - 13.5 |
| Adolescent Boys | 11 – 17 yrs | 12.5 - 16.5 |
| Adolescent Girls | 11 – 17 yrs | 12.0 - 15.5 |
| Adult Men | 18+ yrs | 13.0 - 17.0 |
| Adult Women | 18+ yrs | 12.0 - 15.0 |

## Technologies & Libraries

- **Deep Learning Framework**: TensorFlow / Keras
- **Model Architecture**: EfficientNet-B0 (Pre-trained ImageNet weights)
- **Data Processing**: NumPy, Pandas, OpenCV
- **Image Scaling**: Scikit-learn (MinMaxScaler)
- **Web Interface**: Streamlit
- **Data Management**: Excel (XLSX)
- **Python Version**: 3.8+

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

## Model Architecture

- **Base Model**: EfficientNet-B0
- **Input Size**: 224 x 224 x 3 (RGB images)
- **Transfer Learning**: Pre-trained on ImageNet weights
- **Output**: Continuous hemoglobin value prediction
- **Activation**: ReLU layers with Batch Normalization
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error (MSE)

## File Descriptions

| File/Folder | Purpose |
| :--- | :--- |
| `app.py` | Main Streamlit web application |
| `src/config.py` | Clinical ranges and system configuration |
| `src/data_loader.py` | Dataset loading and preprocessing pipeline |
| `src/model.py` | EfficientNet-B0 model definition |
| `src/utils.py` | Utility functions for classification and inference |
| `notebooks/final_training.ipynb` | Complete training pipeline and experimentation |
| `models/efficientnet_b0_anemia.h5` | Pre-trained model weights |
| `dataset/` | Training data (India & Italy datasets) |

## Key Features of the System

1. **Multi-Region Training**: Model trained on geographically diverse datasets
2. **Age-Specific Classification**: Different thresholds for different age groups
3. **Continuous Prediction**: Outputs precise hemoglobin values, not just binary classification
4. **User-Friendly Interface**: Interactive Streamlit app with real-time predictions
5. **Scalable Architecture**: Can be extended with additional datasets and regions

## Performance Metrics

The model achieves strong performance across validation datasets:
- Handles diverse image qualities and patient demographics
- Provides confidence-based predictions
- Robust classification with clinical accuracy

## Future Enhancements

- [ ] Integration with hospital management systems
- [ ] Mobile application development
- [ ] Multi-language support
- [ ] Advanced statistical reporting
- [ ] Integration with electronic health records (EHR)
- [ ] Continuous model retraining with new data

## Important Disclaimer

This system is designed as a **diagnostic support tool** and should **NOT** be used as a substitute for professional medical advice. Always consult qualified healthcare professionals for accurate diagnosis and treatment decisions. The predictions are based on trained models and should be validated with clinical laboratory tests.

## Troubleshooting

### Common Issues

**Issue**: Module not found errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Port 8501 already in use
- **Solution**: Use custom port: `streamlit run app.py --server.port 8502`

**Issue**: Image upload errors
- **Solution**: Ensure images are in supported formats (JPG, PNG) with reasonable file size

## Contributors

This project was developed with contributions from medical imaging and machine learning experts to improve early detection and diagnosis of anemia.

## License

This project is provided for educational and research purposes. Usage in commercial settings requires appropriate authorization.

## Support & Contact

For issues, questions, or suggestions:
- 📧 Report bugs or feature requests through the project repository
- 🌐 Access the live application: https://anemiaanalysis-toobarani.streamlit.app/
- 📚 Refer to the comprehensive notebooks for implementation details

---

**Last Updated**: March 2026  
**Version**: 1.0.0  
**Status**: Active & Maintained

