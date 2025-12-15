import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Earthquake Prediction Dashboard",
    layout="wide"
)

model = joblib.load("model_gempa.pkl")

st.title("🌍 Earthquake Prediction Dashboard")
st.markdown("Sistem klasifikasi **Gempa vs Tidak Gempa** berbasis Machine Learning")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    X = df[["magnitudo", "kedalaman", "jarak"]]
    preds = model.predict(X)

    df["Prediksi"] = ["GEMPA" if p != 0 else "TIDAK GEMPA" for p in preds]

    st.subheader("📊 Statistik Hasil")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df))
    col2.metric("Gempa", (df["Prediksi"] == "GEMPA").sum())
    col3.metric("Tidak Gempa", (df["Prediksi"] == "TIDAK GEMPA").sum())

    st.subheader("📋 Hasil Prediksi")
    st.dataframe(df)

    st.download_button(
        "⬇️ Download Hasil CSV",
        df.to_csv(index=False),
        file_name="hasil_prediksi_gempa.csv",
        mime="text/csv"
    )
