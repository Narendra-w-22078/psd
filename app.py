import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Earthquake Prediction", layout="wide")

st.title("🌍 Earthquake Prediction System")
st.markdown("Prediksi **Gempa vs Tidak Gempa** berdasarkan **sampel dataset**")

# =========================
# LOAD MODEL & DATASET
# =========================
model = joblib.load("model_gempa.pkl")

df = pd.read_csv("data_preprocessed.csv")

st.subheader("📄 Dataset")
st.dataframe(df.head())

# =========================
# PILIH SAMPEL
# =========================
st.subheader("🎯 Pilih Sampel")

index = st.number_input(
    "Pilih nomor sampel",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

sample = df.iloc[[index]]

st.write("### Data Sampel Terpilih")
st.dataframe(sample)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Sampel"):
    pred = model.predict(sample)[0]

    hasil = "GEMPA" if pred != 0 else "TIDAK GEMPA"

    st.success(f"HASIL PREDIKSI: **{hasil}**")
