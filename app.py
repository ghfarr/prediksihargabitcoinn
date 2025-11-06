import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# Load models
rf_model = joblib.load("model_randomforest.pkl")
lr_model = joblib.load("model_linear.pkl")
lstm_model = load_model("model_lstm.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# ----------- PAGE CONFIG ----------- #
st.set_page_config(page_title="Prediksi BTC", page_icon="💹", layout="centered")

# ----------- CUSTOM CSS UI ----------- #
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap" rel="stylesheet">

<style>
* { font-family: 'Poppins', sans-serif; }

[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #0e0f24 0%, #1c2340 50%, #1a1f33 100%);
  color: white;
}

.card {
  background: rgba(255,255,255,0.06);
  padding: 30px 35px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.2);
  backdrop-filter: blur(14px);
  margin-top: 20px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

h1 {
  text-align: center;
  font-weight: 800;
  font-size: 2.6rem;
  margin-bottom: 5px;
}

.subtitle {
  text-align: center;
  opacity: 0.85;
  margin-bottom: 30px;
}

.stButton>button {
  width: 100%;
  border-radius: 10px;
  background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
  color: white;
  padding: 14px;
  border: none;
  font-weight: 600;
  font-size: 1.05rem;
  transition: 0.3s;
}

.stButton>button:hover {
  transform: scale(1.03);
  box-shadow: 0 0 18px #00d2ff;
}
</style>
""", unsafe_allow_html=True)

# ----------- HEADER ----------- #
st.markdown("<h1>💹 Prediksi Harga Bitcoin (BTC)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Masukkan data untuk memprediksi harga penutupan Bitcoin selanjutnya.</p>", unsafe_allow_html=True)

# ----------- INPUT CARD ----------- #
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    open_price = st.number_input("Harga Open (USD)", min_value=0.0, format="%.2f")
    high_price = st.number_input("Harga High (USD)", min_value=0.0, format="%.2f")
    low_price = st.number_input("Harga Low (USD)", min_value=0.0, format="%.2f")
    volume = st.number_input("Volume Transaksi", min_value=0.0, format="%.2f")
    model_choice = st.selectbox("Pilih Model Prediksi :", ["Random Forest", "Linear Regression", "LSTM"])

    st.markdown("</div>", unsafe_allow_html=True)


# ----------- PREDICT FUNCTION ----------- #
def predict():
    data = np.array([[open_price, high_price, low_price, volume]])
    data_scaled = scaler.transform(data)

    if model_choice == "Random Forest":
        return rf_model.predict(data_scaled)[0]
    elif model_choice == "Linear Regression":
        return lr_model.predict(data_scaled)[0]
    else:
        X = data_scaled.reshape((1, 4, 1))
        return lstm_model.predict(X)[0][0]


# ----------- BUTTON ----------- #
if st.button(" Prediksi Harga BTC"):
    result = predict()
    st.success(f"📈 Perkiraan Harga BTC: **${result:,.2f}**", icon="✅")
