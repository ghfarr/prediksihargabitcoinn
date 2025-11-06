import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model & scaler
rf_model = joblib.load("model_randomforest.pkl")
lr_model = joblib.load("model_linear.pkl")
lstm_model = load_model("model_lstm.h5")
scaler = joblib.load("scaler.pkl")

st.title("📈 Prediksi Harga Bitcoin (BTC)")
st.write("Pilih model Machine Learning untuk melakukan prediksi harga Bitcoin berdasarkan input fitur.")

# Input user
open_price = st.number_input("Harga Open", min_value=0.0, format="%.2f")
high_price = st.number_input("Harga High", min_value=0.0, format="%.2f")
low_price = st.number_input("Harga Low", min_value=0.0, format="%.2f")
volume = st.number_input("Volume Transaksi", min_value=0.0, format="%.2f")

model_choice = st.selectbox("Pilih Model", ["Random Forest", "Linear Regression", "LSTM"])

def predict():
    data = np.array([[open_price, high_price, low_price, volume]])
    data_scaled = scaler.transform(data)

    if model_choice == "Random Forest":
        return rf_model.predict(data_scaled)[0]
    elif model_choice == "Linear Regression":
        return lr_model.predict(data_scaled)[0]
    elif model_choice == "LSTM":
        lstm_input = data_scaled.reshape((1, 1, 4))
        return lstm_model.predict(lstm_input)[0][0]

if st.button("🔮 Prediksi Harga BTC"):
    result = predict()
    st.success(f"Perkiraan Harga Bitcoin : **${result:,.2f}**")
