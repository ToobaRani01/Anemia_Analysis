# EYES-DEFY-ANEMIA — Hemoglobin Prediction System

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

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Training
Use the provided [Jupyter Notebook](notebooks/final_training.ipynb) to train the model. It includes data loading, normalization (MinMaxScaler), and the EfficientNet-B0 training loop.

### 3. Launch App
```bash
streamlit run app.py
```

## Normal Hemoglobin Ranges (g/dL)
Diagnosis is based on the following clinical ranges. Values below the minimum indicate anemia, while values exceeding the maximum are flagged as High/Severe.

| Category | Age Bracket | Normal Range |
| :--- | :--- | :--- |
| Children | 0 – 10 yrs | 11.0 - 13.5 |
| Adolescent Boys | 11 – 17 yrs | 12.5 - 16.5 |
| Adolescent Girls | 11 – 17 yrs | 12.0 - 15.5 |
| Adult Men | 18+ yrs | 13.0 - 17.0 |
| Adult Women | 18+ yrs | 12.0 - 15.0 |

