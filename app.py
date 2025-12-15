import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Earthquake Prediction",
    layout="wide"
)

st.title("🌍 Earthquake Prediction System")
st.markdown("Pilih **sampel data** untuk memprediksi **Gempa / Tidak Gempa**")

# =========================
# LOAD FILE
# =========================
model = joblib.load("model_gempa.pkl")
feature_names = joblib.load("feature_names.pkl")
df = pd.read_csv("data_preprocessed.csv")

# =========================
# TAMPILKAN DATA
# =========================
st.subheader("📄 Dataset (Preprocessed)")
st.dataframe(df.head())

# =========================
# PILIH SAMPEL
# =========================
index = st.number_input(
    "Pilih nomor sampel",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

sample = df.iloc[[index]]
X_sample = sample[feature_names]

st.write("### Sampel Terpilih")
st.dataframe(X_sample)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi"):
    pred = model.predict(X_sample)[0]
    hasil = "GEMPA" if pred != 0 else "TIDAK GEMPA"
    st.success(f"HASIL PREDIKSI: **{hasil}**")
