import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Earthquake Prediction Dashboard",
    layout="wide"
)

st.title("🌍 Earthquake Prediction Dashboard")
st.markdown(
    """
    Sistem klasifikasi **Gempa / Tidak Gempa**  
    berdasarkan **data seismik hasil preprocessing**.
    """
)

# =========================
# LOAD MODEL & DATA
# =========================
model = joblib.load("model_gempa.pkl")
feature_names = joblib.load("feature_names.pkl")
df = pd.read_csv("data_streamlit.csv")

# =========================
# INFO DATASET
# =========================
st.subheader("📊 Informasi Dataset")
col1, col2 = st.columns(2)
col1.metric("Jumlah Sampel", len(df))
col2.metric("Jumlah Fitur", df.shape[1])

st.subheader("📄 Contoh Data")
st.dataframe(df.head())

# =========================
# PILIH SAMPEL
# =========================
st.subheader("🎯 Pilih Sampel Data")

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
if st.button("🔍 Prediksi Sampel"):

    # Prediksi kelas
    pred = model.predict(X_sample)[0]
    hasil = "GEMPA" if pred == 1 else "TIDAK GEMPA"

    # Probabilitas
    proba = model.predict_proba(X_sample)[0]

    # Amplitudo (representatif)
    amplitudo = float(np.max(np.abs(X_sample.values)))

    col1, col2, col3 = st.columns(3)

    col1.success(f"🧭 **Hasil Prediksi**\n\n{hasil}")
    col2.info(f"📈 **Amplitudo**\n\n{amplitudo:.4f}")
    col3.warning(
        f"📊 **Probabilitas Gempa**\n\n{proba[1]*100:.2f}%"
    )

    # =========================
    # VISUALISASI SEISMOMETER
    # =========================
    st.subheader("📉 Visualisasi Seismometer (Representatif)")

    signal = X_sample.values.flatten()
    st.line_chart(signal)

    st.caption(
        "Grafik merupakan visualisasi representatif dari fitur numerik "
        "hasil preprocessing, bukan sinyal mentah seismik."
    )
