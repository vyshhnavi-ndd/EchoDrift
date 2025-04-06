import requests

TOMORROW_API_KEY = "RDskHuYZaZeV6xErtUJCVgjbROFam4fK"  # replace with your Tomorrow.io API key

def get_weather(city, state=None, country=None):
    location = f"{city},{state if state else ''},{country if country else ''}"
    url = f"https://api.tomorrow.io/v4/weather/realtime?location={location}&apikey={TOMORROW_API_KEY}"
    response = requests.get(url)
    data = response.json()

    values = data.get("data", {}).get("values", {})
    return {
        "temperature": values.get("temperature", 0),
        "humidity": values.get("humidity", 0),
        "cloud_cover": values.get("cloudCover", 0),
        "wind_speed": values.get("windSpeed", 0),
        "pressure": values.get("pressureSeaLevel", 0)
    }
