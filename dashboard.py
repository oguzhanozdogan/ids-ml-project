import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from scipy.stats import ks_2samp


st.set_page_config(page_title="IDS Monitoring Dashboard", layout="wide")

st.title("Intrusion Detection Monitoring Dashboard")


# Load reference
try:
    features = load("models/features.pkl")
    train_stats = load("models/train_stats.pkl")
    ref_mean = train_stats["mean"]
    ref_std = train_stats["std"]
except:
    st.error("Train model first to generate reference data")
    st.stop()


uploaded = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)


if uploaded:

    df = pd.read_csv(uploaded)

    st.success("File loaded")

    # --------------------------------
    # OVERVIEW
    # --------------------------------

    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head())

    # --------------------------------
    # SCHEMA CHECK
    # --------------------------------

    st.header("Schema Validation")

    missing = [f for f in features if f not in df.columns]
    extra = [c for c in df.columns if c not in features]

    if not missing and not extra:
        st.success("Schema matches model")
        valid = True
    else:
        st.warning("Schema mismatch")
        st.write("Missing:", missing)
        st.write("Extra:", extra)
        valid = False


    # --------------------------------
    # VISUALIZATION
    # --------------------------------

    st.header("Traffic Distribution")

    num_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    col = st.selectbox(
        "Select Feature",
        num_cols
    )

    st.line_chart(df[col])

    st.bar_chart(df[col].value_counts().head(20))


    # --------------------------------
    # DRIFT DETECTION
    # --------------------------------

    st.header("Data Drift Detection")

    if valid:

        drift_results = []

        for f in features:

            if f not in df.columns:
                continue

            ref_data = np.random.normal(
                ref_mean[f],
                ref_std[f],
                10000
            )

            cur_data = df[f].dropna().values

            if len(cur_data) < 50:
                continue

            stat, pval = ks_2samp(ref_data, cur_data)

            drift = pval < 0.05

            drift_results.append({
                "Feature": f,
                "P-Value": round(pval, 6),
                "Drift": drift
            })

        drift_df = pd.DataFrame(drift_results)

        st.dataframe(drift_df)

        drift_count = drift_df["Drift"].sum()

        st.metric(
            "Drifting Features",
            drift_count
        )

        if drift_count > 0:
            st.error("Data drift detected")
        else:
            st.success("No drift detected")


    # --------------------------------
    # EXPORT
    # --------------------------------

    st.header("Export Clean Data")

    if valid:

        clean_df = df[features].copy()

        clean_df = clean_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        clean_df = clean_df.dropna()

        csv = clean_df.to_csv(index=False).encode()

        st.download_button(
            "Download Clean CIC Data",
            csv,
            "cleaned_cic.csv",
            "text/csv"
        )

    else:

        st.warning(
            "Export disabled. Dataset does not match CIC feature schema."
        )
