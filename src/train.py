import pandas as pd
import numpy as np

from joblib import dump
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight


DATA_PATH = "data/processed/data.csv"
MODEL_PATH = "models/model.pkl"


def main():

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Total rows:", len(df))

    # Use Friday as unseen test day
    test_mask = df["source_file"].str.contains("Friday")

    train_df = df[~test_mask]
    test_df = df[test_mask]

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    # Split features and labels
    X_train = train_df.drop(["Label", "source_file"], axis=1)
    y_train = train_df["Label"]

    X_test = test_df.drop(["Label", "source_file"], axis=1)
    y_test = test_df["Label"]

    # Compute class weights (handle imbalance)
    print("Computing class weights...")

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight = dict(zip(classes, weights))

    scale_pos_weight = class_weight[1] / class_weight[0]

    print("scale_pos_weight:", scale_pos_weight)

    # Train XGBoost
    print("Training model...")

    model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)


    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating...")

    y_prob = model.predict_proba(X_test)[:, 1]

    threshold = 0.3

    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y_test, y_pred, digits=4))

  # Save model and feature schema
    feature_names = X_train.columns.tolist()

    # Save training statistics for drift detection
    train_stats = {
        "mean": X_train.mean().to_dict(),
        "std": X_train.std().to_dict()
    }

    dump(model, MODEL_PATH)
    dump(feature_names, "models/features.pkl")
    dump(train_stats, "models/train_stats.pkl")

    print("\nModel saved to:", MODEL_PATH)
    print("Features saved to: models/features.pkl")
    print("Stats saved to: models/train_stats.pkl")

if __name__ == "__main__":
    main()
