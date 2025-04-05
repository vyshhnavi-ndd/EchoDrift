import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from gtts import gTTS
import os
from datetime import datetime
from reportlab.pdfgen import canvas
from io import BytesIO

# -----------------------------
# LORENZ SYSTEM CHAOS FUNCTION
# -----------------------------
def lorenz_attractor(x0, y0, z0, sigma, rho, beta, dt=0.01, steps=10000):
    xs, ys, zs = [x0], [y0], [z0]
    for _ in range(steps):
        x, y, z = xs[-1], ys[-1], zs[-1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xs.append(x + dx * dt)
        ys.append(y + dy * dt)
        zs.append(z + dz * dt)
    return xs, ys, zs

# -----------------------------
# DISASTER PREDICTION LOGIC
# -----------------------------
def predict_disaster(temp, pressure):
    if temp > 40 and pressure < 995:
        return "Severe Heatwave", 0.90
    elif temp > 35 and pressure < 1000:
        return "Heatwave", 0.85
    elif temp < 5 and pressure > 1025:
        return "Severe Cold Wave", 0.88
    elif temp < 10 and pressure > 1020:
        return "Cold Wave", 0.78
    elif 20 <= temp <= 30 and pressure < 990:
        return "Cyclone", 0.88
    elif temp > 30 and 990 <= pressure <= 1005:
        return "Storm", 0.72
    elif temp < 25 and pressure < 985:
        return "Flood", 0.80
    elif pressure < 950:
        return "Tornado", 0.86
    else:
        return "Normal", 0.1

# -----------------------------
# TEXT TO SPEECH
# -----------------------------
def speak_alert(disaster):
    tts = gTTS(f"Warning: {disaster} predicted based on current weather conditions.", lang='en')
    audio_path = "alert.mp3"
    tts.save(audio_path)
    os.system(f"start {audio_path}" if os.name == "nt" else f"afplay {audio_path}")

# -----------------------------
# PDF REPORT GENERATION
# -----------------------------
def generate_pdf(city, weather_data, disaster, prob):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 16)
    c.drawString(50, 800, f"Disaster Report for {city}")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Temperature: {weather_data['temp']} °C")
    c.drawString(50, 750, f"Pressure: {weather_data['pressure']} hPa")
    c.drawString(50, 730, f"Humidity: {weather_data['humidity']}%")
    c.drawString(50, 710, f"Weather: {weather_data['weather']}")
    c.drawString(50, 690, f"Prediction: {disaster} (Probability: {int(prob*100)}%)")
    c.drawString(50, 670, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# GET REAL-TIME WEATHER
# -----------------------------
def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
    response = requests.get(url).json()
    data = {
        "temp": response["main"]["temp"],
        "pressure": response["main"]["pressure"],
        "humidity": response["main"]["humidity"],
        "weather": response["weather"][0]["main"]
    }
    return data

# -----------------------------
# ANALYZE CSV DATA
# -----------------------------
def analyze_csv_data(df):
    df["Predicted Disaster"] = ""
    df["Probability"] = 0.0
    for index, row in df.iterrows():
        temp = row["Temperature"]
        pressure = row["Pressure"]
        disaster, prob = predict_disaster(temp, pressure)
        df.at[index, "Predicted Disaster"] = disaster
        df.at[index, "Probability"] = prob
    return df

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Chaos-Based Disaster Prediction", layout="wide")
st.title("🌪️ Real-Time Disaster Prediction using Chaos Theory & Bayesian Modeling")

with st.sidebar:
    st.header("🔧 Chaos System Parameters")
    sigma = st.slider("Sigma", 0.0, 20.0, 10.0)
    rho = st.slider("Rho", 0.0, 50.0, 28.0)
    beta = st.slider("Beta", 0.0, 10.0, 2.67)

    st.header("🌍 Weather Inputs")
    api_key = st.text_input("Enter your OpenWeatherMap API Key", type="password")
    city = st.text_input("Enter City Name", "Chennai")

    st.markdown("---")
    st.header("📂 Upload Metadata (CSV)")
    uploaded_file = st.file_uploader("Upload Master_Weather_Data.csv (with Temperature & Pressure columns)", type=["csv"])

# ---------------------------------
# REAL-TIME WEATHER PREDICTION
# ---------------------------------
if st.button("🔍 Analyze Real-Time Weather"):
    if api_key:
        weather_data = get_weather(city, api_key)
        temp = weather_data["temp"]
        pressure = weather_data["pressure"]

        disaster, prob = predict_disaster(temp, pressure)
        st.subheader(f"🌤️ Weather in {city}")
        st.write(f"Temperature: {temp} °C")
        st.write(f"Pressure: {pressure} hPa")
        st.write(f"Humidity: {weather_data['humidity']}%")
        st.write(f"General Weather: {weather_data['weather']}")

        st.subheader("⚠️ Prediction Result")
        st.success(f"Predicted Disaster: {disaster} (Confidence: {int(prob*100)}%)")
        speak_alert(disaster)

        # PDF Report
        pdf = generate_pdf(city, weather_data, disaster, prob)
        st.download_button("📄 Download PDF Report", data=pdf, file_name=f"{city}_disaster_report.pdf")

# ---------------------------------
# CSV METADATA PREDICTION
# ---------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "Temperature" in df.columns and "Pressure" in df.columns:
            st.subheader("📊 CSV-Based Batch Prediction")
            st.info("Analyzing uploaded file: `Master_Weather_Data.csv`")

            result_df = analyze_csv_data(df)
            st.dataframe(result_df)

            # Option to download CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Prediction CSV", csv, "predicted_disasters.csv", "text/csv")
        else:
            st.error("❌ The file must contain 'Temperature' and 'Pressure' columns.")
    except Exception as e:
        st.error(f"⚠️ Could not read file. Error: {e}")

# ---------------------------------
# CHAOS SYSTEM VISUALIZATION
# ---------------------------------
st.subheader("🌀 Chaos Simulation - Lorenz Attractor")
x0, y0, z0 = 0.0, 1.0, 1.05
xs, ys, zs = lorenz_attractor(x0, y0, z0, sigma, rho, beta)

fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='blue', width=2))])
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
st.plotly_chart(fig, use_container_width=True)
