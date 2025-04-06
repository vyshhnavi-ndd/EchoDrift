import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from chaos_model import solve_chaos
from prediction import classify_weather
from voice_output import speak_prediction
from weather_api import get_weather
import pandas as pd

st.set_page_config(page_title="Chaos Weather Predictor", layout="centered")
st.title("🌪️ Real-Time Weather Predictor with Chaos Theory")

mode = st.radio("Choose Weather Input Mode:", ["Manual Input", "Fetch Real-Time Weather"], index=0)

if mode == "Fetch Real-Time Weather":
    st.sidebar.header("📍 Real-time Location Input")
    city = st.sidebar.text_input("City", "Delhi")
    state = st.sidebar.text_input("State (Optional)", "")
    country = st.sidebar.text_input("Country", "India")

    if st.sidebar.button("Fetch Weather Now"):
        real_data = get_weather(city, state, country)
        st.subheader("📡 Real-Time Weather Fetched")
        st.write(real_data)

        st.session_state.real_weather = real_data
    else:
        st.session_state.real_weather = {}

else:
    st.sidebar.header("📝 Manual Weather Input")
    temp_user = st.sidebar.number_input("Temperature (°C)", 0.0, 100.0, 30.0)
    humidity_user = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    cloud_cover_user = st.sidebar.number_input("Cloud Cover (%)", 0.0, 100.0, 20.0)
    wind_speed_user = st.sidebar.number_input("Wind Speed (km/h)", 0.0, 200.0, 10.0)
    pressure_user = st.sidebar.number_input("Pressure (hPa)", 800.0, 1100.0, 1013.0)

    st.session_state.real_weather = {
        "temperature": temp_user,
        "humidity": humidity_user,
        "cloud_cover": cloud_cover_user,
        "wind_speed": wind_speed_user,
        "pressure": pressure_user
    }

st.sidebar.header("⚙️ Chaos Parameters")
sigma = st.sidebar.slider("σ (Sigma)", 0.0, 20.0, 10.0)
rho = st.sidebar.slider("ρ (Rho)", 0.0, 50.0, 28.0)
beta = st.sidebar.slider("β (Beta)", 0.0, 10.0, 2.666)
d0 = st.sidebar.slider("d₀", 0.0, 10.0, 0.5)

real = st.session_state.real_weather
if real and mode == "Fetch Real-Time Weather":
    st.subheader("🔍 Difference Between Real-time & Manual Defaults")
    df_diff = pd.DataFrame({
        "Parameter": ["Temperature", "Humidity", "Cloud Cover", "Wind Speed", "Pressure"],
        "Real-time": [real["temperature"], real["humidity"], real["cloud_cover"], real["wind_speed"], real["pressure"]],
        "Default Manual": [30.0, 60.0, 20.0, 10.0, 1013.0]
    })
    st.table(df_diff)

# Chaos Model
state0 = [1.0, 1.0, 1.0, 0.1, 0.1]
t = np.linspace(0, 30, 3000)
solution = solve_chaos(state0, t, sigma, rho, beta, d0)
x, y, z, y1, z1 = solution.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
st.pyplot(fig)

# Classification + Audio
result, icon = classify_weather(z, z1)
st.subheader("📊 Prediction Result")
st.success(result)
st.image(f"assets/{icon}")

speak_prediction(result)
audio_file = open("prediction.mp3", "rb")
st.audio(audio_file.read(), format='audio/mp3')

# Report download
df_report = pd.DataFrame({
    "Sigma": [sigma],
    "Rho": [rho],
    "Beta": [beta],
    "Prediction": [result]
})
st.download_button("📄 Download Report", df_report.to_csv(index=False), file_name="chaos_weather_report.csv")
