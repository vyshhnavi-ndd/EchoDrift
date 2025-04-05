import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gtts import gTTS
import os
import requests
from io import BytesIO
from PIL import Image
from fpdf import FPDF
import re
from sklearn.naive_bayes import GaussianNB

# -------------------------------
# Lorenz Attractor System
# -------------------------------
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# -------------------------------
# Helper to remove emojis / Unicode (for PDF safety)
# -------------------------------
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# -------------------------------
# Real-Time Weather API Function
# -------------------------------
def get_weather(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind": data["wind"]["speed"],
            "description": data["weather"][0]["description"]
        }
    else:
        return None

# -------------------------------
# Bayesian Classifier for Prediction
# -------------------------------
def bayesian_prediction(temp, pressure):
    # Normalized training data (temperature °C, pressure hPa) => [Clear, Rain, Storm]
    X = np.array([
        [20, 1015],  # Clear
        [25, 1008],  # Rain
        [30, 998],   # Rain
        [35, 990],   # Storm
        [15, 1022],  # Clear
        [32, 985]    # Storm
    ])
    y = np.array(["Clear", "Rain", "Rain", "Storm", "Clear", "Storm"])

    model = GaussianNB()
    model.fit(X, y)

    prediction = model.predict([[temp, pressure]])[0]
    return prediction

# -------------------------------
# Streamlit Web App UI
# -------------------------------
st.set_page_config(page_title="Chaos-Based Weather Predictor", layout="wide")
st.title("🌩️ Chaos-Based Weather Prediction Dashboard")
st.markdown("🔬 Powered by Lorenz Attractor + Live Weather")

# Real-Time Weather Section
st.subheader("🌦️ Real-Time Weather Input")
city = st.text_input("Enter City Name", "Chennai")
api_key = "de2f9dedbad308fd7638a186797cedf5"

if city:
    weather = get_weather(city, api_key)
    if weather:
        st.success("✅ Weather Data Fetched Successfully!")
        st.write(f"**🌡️ Temperature:** {weather['temp']} °C")
        st.write(f"**💧 Humidity:** {weather['humidity']} %")
        st.write(f"**🔵 Pressure:** {weather['pressure']} hPa")
        st.write(f"**🌬️ Wind Speed:** {weather['wind']} m/s")
        st.write(f"**📡 Condition:** {weather['description'].title()}")
    else:
        st.warning("⚠️ Could not fetch weather. Check city name or API key.")

# -------------------------------
# Chaos Control Panel
# -------------------------------
st.markdown("### 🌀 Lorenz Chaos Simulation")
col1, col2 = st.columns(2)

if weather:
    auto_sigma = round(np.interp(weather['temp'], [0, 45], [0.1, 25.0]), 2)
    auto_rho = round(np.interp(weather['humidity'], [0, 100], [0.1, 50.0]), 2)
    auto_beta = round(np.interp(weather['pressure'], [950, 1050], [0.1, 10.0]), 2)

    with col1:
        sigma = st.slider("σ (Sigma)", 0.1, 25.0, auto_sigma)
        rho = st.slider("ρ (Rho)", 0.1, 50.0, auto_rho)
        beta = st.slider("β (Beta)", 0.1, 10.0, auto_beta)
        duration = st.slider("Simulation Time", 10, 100, 50)

    with col2:
        st.success("🎯 Chaos parameters mapped from weather:")
        st.write(f"**Mapped σ (Sigma):** {auto_sigma}")
        st.write(f"**Mapped ρ (Rho):** {auto_rho}")
        st.write(f"**Mapped β (Beta):** {auto_beta}")
else:
    with col1:
        sigma = st.slider("σ (Sigma)", 0.1, 25.0, 10.0)
        rho = st.slider("ρ (Rho)", 0.1, 50.0, 28.0)
        beta = st.slider("β (Beta)", 0.1, 10.0, 2.67)
        duration = st.slider("Simulation Time", 10, 100, 50)

    with col2:
        st.warning("⛅ Default chaos values used (no weather data)")
        st.image("https://i.gifer.com/9oW.gif", width=300)

# -------------------------------
# Lorenz Simulation
# -------------------------------
state0 = [0.1, 0.0, 0.0]
t = np.linspace(0, duration, 10000)
solution = odeint(lorenz, state0, t, args=(sigma, rho, beta))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], lw=0.5, color='purple')
ax.set_title("Lorenz Attractor")

st.pyplot(fig)

# Convert plot to downloadable image
buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.download_button("📥 Download Lorenz Plot", buf, file_name="lorenz_plot.png", mime="image/png")

# -------------------------------
# Prediction Logic with Bayesian Model
# -------------------------------
st.subheader("🔮 Prediction Based on Chaos + Bayesian Model")
if weather:
    bayes_result = bayesian_prediction(weather['temp'], weather['pressure'])
    if sigma > 15 and rho > 30 and bayes_result == "Storm":
        prediction = "⚡ Severe Storm Expected (Bayesian Confirmed)"
    elif beta < 2 and bayes_result == "Clear":
        prediction = "🌤️ Clear Weather (Bayesian Confirmed)"
    elif bayes_result == "Rain":
        prediction = "🌧️ Possible Rain (Bayesian Supported)"
    else:
        prediction = f"🤔 Uncertain: Bayesian says '{bayes_result}'"
else:
    prediction = "No Weather Data to Predict"

st.subheader(f"Prediction: {prediction}")

# -------------------------------
# Text-to-Speech Output
# -------------------------------
if st.button("🔊 Speak Prediction"):
    tts = gTTS(prediction)
    tts.save("prediction.mp3")
    os.system("start prediction.mp3")  # Windows only

# -------------------------------
# Multi-City Chaos Comparison
# -------------------------------
st.markdown("---")
st.markdown("## 🌍 Multi-City Chaos Comparison")

multi_cities = st.text_input("Enter cities (comma separated)", "Chennai, Delhi, Mumbai")

if st.button("Compare Chaos Paths"):
    fig_multi = plt.figure(figsize=(12, 8))
    ax_multi = fig_multi.add_subplot(111, projection='3d')

    for city_name in [c.strip() for c in multi_cities.split(',') if c.strip()]:
        weather_data = get_weather(city_name, api_key)
        if weather_data:
            sigma_c = round(np.interp(weather_data['temp'], [0, 45], [0.1, 25.0]), 2)
            rho_c = round(np.interp(weather_data['humidity'], [0, 100], [0.1, 50.0]), 2)
            beta_c = round(np.interp(weather_data['pressure'], [950, 1050], [0.1, 10.0]), 2)
            sol = odeint(lorenz, state0, t, args=(sigma_c, rho_c, beta_c))
            ax_multi.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=0.5, label=city_name)

    ax_multi.set_title("Multi-City Chaos Paths")
    ax_multi.legend()
    st.pyplot(fig_multi)

# -------------------------------
# Report PDF Export
# -------------------------------
st.markdown("---")
st.subheader("📄 Export Report")

if st.button("📤 Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title(clean_text("Chaos-Based Weather Report"))

    pdf.cell(200, 10, txt=clean_text("Chaos-Based Weather Report"), ln=1, align="C")
    pdf.cell(200, 10, txt=clean_text(f"City: {city}"), ln=2)

    if weather:
        pdf.cell(200, 10, txt=clean_text(f"Temperature: {weather['temp']}°C"), ln=3)
        pdf.cell(200, 10, txt=clean_text(f"Humidity: {weather['humidity']}%"), ln=4)
        pdf.cell(200, 10, txt=clean_text(f"Pressure: {weather['pressure']} hPa"), ln=5)
        pdf.cell(200, 10, txt=clean_text(f"Wind Speed: {weather['wind']} m/s"), ln=6)
        pdf.cell(200, 10, txt=clean_text(f"Condition: {weather['description'].title()}"), ln=7)
        pdf.cell(200, 10, txt=clean_text(f"Chaos Parameters - Sigma: {sigma}, Rho: {rho}, Beta: {beta}"), ln=8)
        pdf.cell(200, 10, txt=clean_text(f"Prediction: {prediction}"), ln=9)

        image_path = "lorenz_image.png"
        fig.savefig(image_path)
        pdf.image(image_path, x=10, y=None, w=180)

    pdf_path = "weather_chaos_report.pdf"
    pdf.output(pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name=pdf_path)
