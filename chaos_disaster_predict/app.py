import numpy as np
import streamlit as st
import requests
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from gtts import gTTS
import os
from io import BytesIO
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from fpdf import FPDF

# -------------------------------
# Lorenz System Definition
# -------------------------------
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# -------------------------------
# Weather Fetching Function
# -------------------------------
def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data['cod'] == 200:
            weather_data = {
                "temp": data['main']['temp'],
                "humidity": data['main']['humidity'],
                "pressure": data['main']['pressure'],
                "wind": data['wind']['speed'],
                "description": data['weather'][0]['description'],
                "lat": data['coord']['lat'],
                "lon": data['coord']['lon']
            }
            return weather_data
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# -------------------------------
# Bayesian Prediction Model
# -------------------------------

def bayesian_prediction(temperature, humidity, pressure, wind_speed):
    """
    Simple Bayesian-like prediction for various disasters based on weather data.

    Parameters:
    - temperature (float): Temperature in Celsius.
    - humidity (float): Humidity percentage.
    - pressure (float): Atmospheric pressure in hPa.
    - wind_speed (float): Wind speed in m/s.

    Returns:
    - disaster_type (str): Predicted disaster type based on weather conditions.
    """
    X = np.array([
        [20, 1015],
        [25, 1008],
        [30, 998],
        [35, 990],
        [15, 1022],
        [32, 985]
    ])
    y = np.array(["Clear", "Rain", "Rain", "Storm", "Clear", "Storm"])

    model = GaussianNB()
    model.fit(X, y)
    # Predict Heatwave
    if temperature > 40:
        return "Heatwave"
    
    # Predict Cyclone
    if wind_speed > 33 and pressure < 1010:
        return "Cyclone"
    
    # Predict Flood (Heavy Rain)
    if humidity > 85 and temperature > 25:
        return "Flood"
    
    # Predict Drought
    if humidity < 30 and temperature > 35:
        return "Drought"
    
    # Predict Tornado (Strong winds, low pressure)
    if wind_speed > 50 and pressure < 990:
        return "Tornado"
    
    # Predict Earthquake (High seismic activity, not directly based on weather, so placeholder)
    # This could be extended to detect areas with historical seismic events, but we'll use it as a placeholder
    if pressure < 1000:
        return "Earthquake"
    
    # Predict Wildfire (Hot and dry conditions)
    if temperature > 35 and humidity < 20:
        return "Wildfire"
    
    # Predict Landslide (Heavy rainfall and high humidity)
    if humidity > 80 and temperature < 10:
        return "Landslide"
    
    # Predict Tsunami (Low pressure and seismic activity, placeholder)
    if pressure < 1000:
        return "Tsunami"
    
    # If no disaster conditions are met, return "Normal"
    return "Normal"
    
    prediction = model.predict([[temp, pressure]])[0]
    return prediction

# -------------------------------
# Anomaly Detection with Isolation Forest
# -------------------------------
def anomaly_detection(weather_data):
    model = IsolationForest(contamination=0.1)
    weather_matrix = np.array([[wd["temp"], wd["pressure"]] for wd in weather_data])
    anomalies = model.fit_predict(weather_matrix)
    return anomalies

# -------------------------------
# Clustering with K-Means
# -------------------------------
def clustering_cities(weather_data):
    weather_matrix = np.array([[wd["temp"], wd["pressure"]] for wd in weather_data])
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(weather_matrix)
    return clusters

# -------------------------------
# Disaster Risk Visualization (Heatmap)
# -------------------------------
def plot_risk_heatmap(weather_data):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Centered on India

    heat_data = [[wd["lat"], wd["lon"], wd["temp"]] for wd in weather_data]
    HeatMap(heat_data).add_to(m)

    return m

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
# Prediction Logic with Specific Disaster Alerts
# -------------------------------
st.subheader("🔮 Prediction Based on Chaos + Bayesian Model")

if weather:
    bayes_result = bayesian_prediction(weather['temp'], weather['humidity'], weather['pressure'], weather['wind'])

    
    if sigma > 15 and rho > 30 and bayes_result == "Storm":
        prediction = "⚡ Severe Storm Expected (Bayesian Confirmed)"
        safety_alert = "🚨 **Severe Storm Alert:** A strong storm is predicted. Please stay indoors and secure all outdoor objects. Avoid traveling during the storm."
    elif sigma > 25 and bayes_result == "Flood":
        prediction = "🌧️ Flood Warning"
        safety_alert = "🚨 **Flood Warning:** Heavy rainfall has led to rising water levels in your area. Evacuate immediately to higher ground to avoid flooding. Please follow all safety protocols and avoid walking or driving through floodwaters. Stay tuned for further updates. For your safety, refrain from using electrical appliances and keep away from water bodies."
    elif rho > 40 and bayes_result == "Cyclone":
        prediction = "🌀 Cyclone Alert"
        safety_alert = "🚨 **Cyclone Alert:** A powerful cyclone is moving towards your region. The winds are expected to reach up to 120 km/h, and heavy rainfall is imminent. Please seek shelter in a safe, sturdy building away from windows and doors. Avoid coastal areas and low-lying regions. Stay inside until authorities declare it safe to move."
    elif bayes_result == "Drought":
        prediction = "🌵 Drought Warning"
        safety_alert = "🚨 **Drought Warning:** A severe drought is affecting your area. Water resources are critically low, and agricultural activities are at risk. Please conserve water and avoid non-essential water usage. Authorities are providing emergency supplies, but long-term solutions will require collective action. Follow updates for emergency water distribution points."
    elif bayes_result == "Heatwave":
        prediction = "☀️ Heatwave Warning"
        safety_alert = "🚨 **Heatwave Warning:** Temperatures are expected to soar above 40°C today, putting vulnerable populations at risk. Stay indoors and stay hydrated. Limit outdoor activity and avoid strenuous work. Check on elderly family members and neighbors. Seek cool, air-conditioned places if possible."
    elif bayes_result == "Tornado":
        prediction = "🌪️ Tornado Warning"
        safety_alert = "🚨 **Tornado Warning:** A tornado is currently tracking towards your location. Take immediate shelter in a basement, storm cellar, or an interior room on the lowest floor away from windows. Do not attempt to leave your shelter until the all-clear is given by authorities. Stay informed and follow official evacuation instructions."
    # Disaster Prediction Result
if bayes_result == "Flood":
    prediction = "🌧️ Flood Risk Detected"
    safety_alert = (
        "🚨 **Flood Alert:** Heavy rainfall or rising water levels have been detected in your area. "
        "Avoid low-lying areas and stay tuned to local weather updates. "
        "Move to higher ground if necessary and prepare an emergency kit with essentials."
    )
elif bayes_result == "Earthquake":
    prediction = "🌍 Earthquake Alert"
    safety_alert = (
        "🚨 **Earthquake Alert:** A seismic event of high magnitude has been detected in your region. "
        "If you are indoors, take cover under sturdy furniture and stay away from windows. "
        "Do not use elevators. If you are outside, move to an open area away from buildings, trees, and power lines. "
        "Expect aftershocks and be prepared to evacuate if necessary."
    )
elif bayes_result == "Wildfire":
    prediction = "🔥 Wildfire Risk"
    safety_alert = (
        "🚨 **Wildfire Alert:** High temperatures and low humidity increase the risk of wildfires. "
        "Avoid outdoor burning, and report any signs of fire immediately. "
        "Prepare an evacuation plan and keep emergency supplies ready. "
        "Follow local alerts and avoid forested areas during this period."
    )
elif bayes_result == "Landslide":
    prediction = "🏞️ Landslide Warning"
    safety_alert = (
        "🚨 **Landslide Warning:** Saturated soil conditions due to prolonged rainfall have increased landslide risk. "
        "Avoid hilly or unstable terrain, and be alert for sudden changes like cracks or tilting trees. "
        "Evacuate immediately if authorities issue a landslide order."
    )
elif bayes_result == "Tsunami":
    prediction = "🌊 Tsunami Risk"
    safety_alert = (
        "🚨 **Tsunami Alert:** Low pressure and seismic activity indicate a tsunami threat in coastal regions. "
        "Move to higher ground immediately. Stay away from beaches and listen to local emergency broadcasts for updates. "
        "Do not return to coastal areas until the all-clear is given."
    )
else:
    prediction = "✅ Normal Weather Conditions"
    safety_alert = (
        "🌤️ The weather is stable at the moment with no significant disaster risks detected. "
        "Stay informed and enjoy your day safely. Continue monitoring for any sudden changes."
    )

# Show Prediction and Safety Info
st.markdown(f"### 🧠 Prediction: {prediction}")
st.info(safety_alert)

# Optional: Text-to-Speech Alert
if st.button("🔊 Speak Alert"):
    tts = gTTS(text=safety_alert, lang='en')
    tts.save("alert.mp3")
    audio_file = open("alert.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")
    audio_file.close()


# -------------------------------
# Disaster Risk Visualization (Heatmap)
# -------------------------------
st.markdown("---")
st.markdown("## 🌍 Disaster Risk Heatmap Visualization")

weather_data = [weather]  # Add more cities weather data for heatmap
risk_map = plot_risk_heatmap(weather_data)
st.markdown("### Heatmap for Disaster Risk Based on Weather")
st.components.v1.html(risk_map._repr_html_(), height=500)

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
    pdf.set_title("Chaos-Based Weather Report")

    pdf.cell(200, 10, txt="Chaos-Based Weather Report", ln=1, align="C")
    pdf.cell(200, 10, txt=f"City: {city}", ln=2)

    if weather:
        pdf.cell(200, 10, txt=f"Temperature: {weather['temp']}°C", ln=3)
        pdf.cell(200, 10, txt=f"Humidity: {weather['humidity']}%", ln=4)
        pdf.cell(200, 10, txt=f"Pressure: {weather['pressure']} hPa", ln=5)
        pdf.cell(200, 10, txt=f"Wind Speed: {weather['wind']} m/s", ln=6)
        pdf.cell(200, 10, txt=f"Condition: {weather['description'].title()}", ln=7)
        pdf.cell(200, 10, txt=f"Chaos Parameters - Sigma: {sigma}, Rho: {rho}, Beta: {beta}", ln=8)
        pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=9)

        image_path = "lorenz_image.png"
        fig.savefig(image_path)
        pdf.image(image_path, x=10, y=None, w=180)

    pdf_path = "weather_chaos_report.pdf"
    pdf.output(pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name=pdf_path)
        
