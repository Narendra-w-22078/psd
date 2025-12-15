import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Earthquake Prediction Dashboard",
    layout="wide"
)

st.title("🌍 Earthquake Prediction Dashboard")
st.markdown(
    "Prediksi **Gempa / Tidak Gempa** disertai **Amplitudo** dan "
    "**Visualisasi Seismometer (representatif)**"
)

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
st.subheader("🎯 Pilih Sampel")

index = st.number_input(
    "Pilih nomor sampel",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

sample = df.iloc[[index]]
X_sample = sample[feature_names]

st.write("### Data Sampel Terpilih")
st.dataframe(X_sample)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Sampel"):

    pred = model.predict(X_sample)[0]
    hasil = "GEMPA" if pred != 0 else "TIDAK GEMPA"

    # =========================
    # HITUNG AMPLITUDO (REPRESENTATIF)
    # =========================
    amplitudo = float(np.max(np.abs(X_sample.values)))

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"### 🧭 Hasil Prediksi\n**{hasil}**")

    with col2:
        st.info(f"### 📈 Amplitudo\n**{amplitudo:.4f}**")

    # =========================
    # VISUALISASI SEISMOMETER
    # =========================
    st.subheader("📊 Visualisasi Seismometer (Representatif)")

    signal = X_sample.values.flatten()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, linewidth=1)
    ax.axhline(0, linestyle="--", alpha=0.5)

    ax.set_title("Seismogram (Representasi Fitur Sampel)")
    ax.set_xlabel("Index Fitur")
    ax.set_ylabel("Amplitudo")

    st.pyplot(fig)

    st.caption(
        "Catatan: Grafik ini merupakan visualisasi representatif "
        "berdasarkan fitur hasil preprocessing, bukan sinyal mentah seismik."
    )
