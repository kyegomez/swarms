import json
import requests
from typing import Dict, Any


def fetch_weather_data(city: str) -> Dict[str, Any]:
    """
    Fetch near real-time weather data for a city using wttr.in.

    Args:
        city (str): The name of the city (e.g., "Austin, Tx").

    Returns:
        Dict[str, Any]: Weather data for the specified city.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    url = f"http://wttr.in/{city}"
    params = {"format": "j1"}  # JSON format
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        response = json.dumps(response.json(), indent=2)
        return response
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch weather data: {e}")
    except ValueError:
        raise Exception("Invalid response format.")


# # Example usage
# city = "Huntsville, AL"

# try:
#     weather_data = fetch_weather_data(city)
#     print("Weather Data:", weather_data)
# except Exception as e:
#     print(e)
