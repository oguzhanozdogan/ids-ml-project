# Intrusion Detection System with Machine Learning

This project implements a production-style Network Intrusion Detection System
using machine learning trained on the CIC-IDS2017 dataset.

The system includes data processing, model training, REST API deployment,
Docker containerization, and real-time monitoring.

---

## Features

- XGBoost-based intrusion detection
- Trained on 2.8M+ network flows
- Temporal train/test split
- Class imbalance handling
- FastAPI REST API
- Docker deployment
- Streamlit monitoring dashboard
- Data drift detection
- Schema validation
- Traffic visualization

---

## Project Structure

ids-ml-project/
- data/
  - raw/
  - processed/
- models/
- src/
  - merge_data.py
  - train.py
  - api.py
- dashboard.py
- Dockerfile
- requirements.txt
- README.md

---

## Installation

Create virtual environment:

python -m venv venv  
source venv/Scripts/activate

Install dependencies:

pip install -r requirements.txt

---

## Data Preparation

Merge raw CSV files:

python src/merge_data.py

Output:

data/processed/data.csv

---

## Model Training

Train model and generate reference statistics:

python src/train.py

Output:

models/model.pkl  
models/features.pkl  
models/train_stats.pkl

---

## Run API

Local:

uvicorn src.api:app --reload

Docker:

docker build -t ids-api .  
docker run -p 8000:8000 ids-api

API docs:

http://localhost:8000/docs

---

## Run Dashboard

streamlit run dashboard.py

Open:

http://localhost:8501

Features:

- Data quality analysis
- Schema validation
- Drift detection
- Traffic visualization
- Clean data export

---

## Testing API

Generate valid input:

python

import pandas as pd  
import json  

df = pd.read_csv("data/processed/data.csv")  
row = df.drop(["Label","source_file"], axis=1).iloc[0]  

print(json.dumps({"data":[row.to_dict()]}, indent=2))

Paste into Swagger UI.

---

## Technologies

- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- FastAPI
- Docker
- Streamlit
- SciPy

---

## Author

Oguzhan Özdogan