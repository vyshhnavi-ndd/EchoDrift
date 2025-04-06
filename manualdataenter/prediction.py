def classify_weather(z, z1):
    max_z = max(z)
    max_z1 = max(z1)
    
    if max_z > 45 and max_z1 < 15:
        return "🔥 Heatwave Detected", "heatwave.gif"
    elif max_z < 15 and max_z1 > 25:
        return "❄️ Cold Wave Likely", "coldwave.gif"
    elif 20 <= max_z <= 30 and max_z1 > 30:
        return "🌀 Cyclone Risk", "cyclone.gif"
    elif max_z > 30 and 15 <= max_z1 <= 25:
        return "🌬️ Storm Alert", "storm.gif"
    elif max_z < 25 and max_z1 < 10:
        return "🌊 Flood Possible", "flood.gif"
    else:
        return "✅ No Disaster Detected", "sun.gif"
