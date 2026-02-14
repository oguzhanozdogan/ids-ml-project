from fastapi import FastAPI, HTTPException
import pandas as pd
from joblib import load


app = FastAPI(title="IDS ML API")


MODEL_PATH = "models/model.pkl"
FEATURE_PATH = "models/features.pkl"

model = load(MODEL_PATH)
features = load(FEATURE_PATH)


@app.get("/")
def home():
    return {"status": "IDS API running"}


@app.post("/predict")
def predict(payload: dict):

    if "data" not in payload:
        raise HTTPException(400, "Missing 'data' field")

    df = pd.DataFrame(payload["data"])

    # Check missing features
    missing = [f for f in features if f not in df.columns]

    if missing:
        raise HTTPException(
            400,
            f"Missing required features: {missing}"
        )

    # Reorder columns
    df = df[features]

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isnull().any().any():
        raise HTTPException(
            400,
            "Invalid or non-numeric values detected"
        )

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = []

    for p, r in zip(preds, probs):
        results.append({
            "prediction": int(p),
            "risk": float(r)
        })

    return {"results": results}
