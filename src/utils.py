def aqi_category(aqi):
    if aqi <= 50: return "Good 😊"
    elif aqi <= 100: return "Moderate 😐"
    elif aqi <= 150: return "Unhealthy for Sensitive 😷"
    elif aqi <= 200: return "Unhealthy 🚨"
    elif aqi <= 300: return "Very Unhealthy ☠️"
    else: return "Hazardous 🔴"
