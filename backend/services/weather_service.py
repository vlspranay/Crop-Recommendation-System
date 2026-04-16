import requests

API_KEY = "986a76826e01e568a939b2b63a377aa9"


def get_weather_data(lat, lon):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
            f"&appid={API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=10)
        data = res.json()

        rainfall = data.get("rain", {}).get("1h", 0) * 24  # approximate daily rainfall
        temperature = data["main"]["temp"]

        return {
            "rainfall": rainfall,
            "temperature": temperature,
        }

    except Exception as e:
        print("Weather API error:", e)
        return None
