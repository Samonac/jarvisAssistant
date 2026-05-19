"""Weather Client for Jarvis Assistant.

Fetches weather data from WeatherAPI.com (free tier: 1M calls/month).
Requires a free API key from https://www.weatherapi.com/signup.aspx
Supports current weather and 3-day forecast by city name.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15


class WeatherClient:
    """Fetches current weather and forecasts from WeatherAPI.com.

    Free tier: 1,000,000 calls/month, no credit card required.
    Get a key at: https://www.weatherapi.com/signup.aspx

    Attributes:
        api_key: WeatherAPI.com API key.
        default_city: Default city for weather queries.
    """

    BASE_URL = "https://api.weatherapi.com/v1"

    def __init__(self, api_key: str, default_city: str = "Paris"):
        self.api_key = api_key
        self.default_city = default_city

    def is_configured(self) -> bool:
        """Check if the weather API key is set."""
        return bool(self.api_key)

    def get_current(self, city: Optional[str] = None) -> dict:
        """Get current weather for a city.

        Args:
            city: City name, zip code, or 'lat,lon'. Uses default_city if not provided.

        Returns:
            Dict with weather data or 'error' key on failure.
        """
        city = city or self.default_city
        try:
            resp = requests.get(
                f"{self.BASE_URL}/current.json",
                params={"key": self.api_key, "q": city, "aqi": "no"},
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 401:
                return {"error": "Invalid WeatherAPI key. Please check your WEATHER_API_KEY."}
            if resp.status_code == 400:
                return {"error": f"City '{city}' not found."}
            if resp.status_code >= 400:
                return {"error": f"Weather API error: {resp.status_code}"}

            data = resp.json()
            loc = data.get("location", {})
            cur = data.get("current", {})
            cond = cur.get("condition", {})

            return {
                "city": loc.get("name", city),
                "country": loc.get("country", ""),
                "region": loc.get("region", ""),
                "temperature": cur.get("temp_c"),
                "feels_like": cur.get("feelslike_c"),
                "humidity": cur.get("humidity"),
                "description": cond.get("text", ""),
                "wind_speed": cur.get("wind_kph"),
                "wind_dir": cur.get("wind_dir", ""),
                "cloud_cover": cur.get("cloud"),
                "visibility": cur.get("vis_km"),
                "uv_index": cur.get("uv"),
                "is_day": cur.get("is_day", 1),
                "local_time": loc.get("localtime", ""),
            }

        except requests.exceptions.Timeout:
            return {"error": "Weather service timed out."}
        except requests.exceptions.ConnectionError:
            return {"error": "Could not connect to weather service."}
        except Exception as e:
            logger.error("Weather error: %s", e)
            return {"error": f"Weather error: {e}"}

    def get_forecast(self, city: Optional[str] = None, days: int = 3) -> dict:
        """Get weather forecast for the next N days (max 3 on free tier).

        Args:
            city: City name. Uses default_city if not provided.
            days: Number of days (1-3 on free tier).

        Returns:
            Dict with forecast data or 'error' key on failure.
        """
        city = city or self.default_city
        days = max(1, min(days, 3))  # Free tier caps at 3 days

        try:
            resp = requests.get(
                f"{self.BASE_URL}/forecast.json",
                params={
                    "key": self.api_key,
                    "q": city,
                    "days": days,
                    "aqi": "no",
                    "alerts": "no",
                },
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 401:
                return {"error": "Invalid WeatherAPI key."}
            if resp.status_code == 400:
                return {"error": f"City '{city}' not found."}
            if resp.status_code >= 400:
                return {"error": f"Forecast API error: {resp.status_code}"}

            data = resp.json()
            loc = data.get("location", {})
            forecast_days = data.get("forecast", {}).get("forecastday", [])

            forecasts = []
            for day in forecast_days:
                d = day.get("day", {})
                cond = d.get("condition", {})
                astro = day.get("astro", {})
                forecasts.append({
                    "date": day.get("date", ""),
                    "temp_max": d.get("maxtemp_c"),
                    "temp_min": d.get("mintemp_c"),
                    "avg_temp": d.get("avgtemp_c"),
                    "description": cond.get("text", ""),
                    "precipitation": d.get("totalprecip_mm", 0),
                    "humidity": d.get("avghumidity"),
                    "wind_max": d.get("maxwind_kph"),
                    "sunrise": astro.get("sunrise", ""),
                    "sunset": astro.get("sunset", ""),
                    "chance_of_rain": d.get("daily_chance_of_rain", 0),
                })

            return {
                "city": loc.get("name", city),
                "country": loc.get("country", ""),
                "forecasts": forecasts,
            }

        except requests.exceptions.Timeout:
            return {"error": "Forecast service timed out."}
        except requests.exceptions.ConnectionError:
            return {"error": "Could not connect to forecast service."}
        except Exception as e:
            logger.error("Forecast error: %s", e)
            return {"error": f"Forecast error: {e}"}
