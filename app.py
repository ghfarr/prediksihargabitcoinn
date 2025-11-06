import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# Load models
rf_model = joblib.load("model_randomforest.pkl")
lr_model = joblib.load("model_linear.pkl")
lstm_model = load_model("model_lstm.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# ---------------------- UI STYLE ----------------------
st.set_page_config(page_title="Prediksi Harga Bitcoin", page_icon="💹", layout="centered")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0f24 0%, #1b263b 100%);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.title {
    color: white;
    text-align: center;
    font-size: 38px;
    font-weight: 800;
}
.card {
    background: #ffffff10;
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(12px);
    border: 1px solid #ffffff30;
}
label, h3, p, span {
    color: white !important;
}
.stButton>button {
    background-color: #33c4ff;
    color: black;
    border-radius: 8px;
    font-weight: bold;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stButton>button:hover {
    background-color: #6ee8ff;
    border: none;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------- UI CONTENT ----------------------
st.markdown("<h1 class='title'>💹 Prediksi Harga Bitcoin (BTC)</h1>", unsafe_allow_html=True)
st.write("Masukkan data untuk memprediksi harga penutupan Bitcoin berikutnya.")

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    open_price = st.number_input("Harga Open", min_value=0.0, format="%.2f")
    high_price = st.number_input("Harga High", min_value=0.0, format="%.2f")
    low_price = st.number_input("Harga Low", min_value=0.0, format="%.2f")
    volume = st.number_input("Volume Transaksi", min_value=0.0, format="%.2f")

    model_choice = st.selectbox("Pilih Model", ["Random Forest", "Linear Regression", "LSTM"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PREDICT ----------------------
def predict():
    data = np.array([[open_price, high_price, low_price, volume]])
    data_scaled = scaler.transform(data)

    if model_choice == "Random Forest":
        return rf_model.predict(data_scaled)[0]
    elif model_choice == "Linear Regression":
        return lr_model.predict(data_scaled)[0]
    else:
        X = data_scaled.reshape((data_scaled.shape[0], data_scaled.shape[1], 1))
        return lstm_model.predict(X)[0][0]

if st.button("🔮 Prediksi Harga BTC"):
    result = predict()
    st.success(f"📈 Perkiraan Harga BTC: **${result:,.2f}**", icon="✅")
