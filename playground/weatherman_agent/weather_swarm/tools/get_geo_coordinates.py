import requests
from typing import List, Dict, Any


def fetch_geocode_by_city(
    api_key: str, city: str, timestamp: int, signature: str
) -> List[Dict[str, Any]]:
    """
    Fetch geocode data by city name.

    Args:
        api_key (str): The API key for authentication.
        city (str): The name of the city (e.g., "Austin, Tx").
        timestamp (int): The timestamp for the request.
        signature (str): The signature for the request.

    Returns:
        List[Dict[str, Any]]: Geocode data for the specified city.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    url = f"https://api.velocityweather.com/v1/{api_key}/reports/geocode/city.json"
    params = {"name": city, "ts": timestamp, "sig": signature}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("geocode", {}).get("data", [])
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch geocode data by city: {e}")
    except ValueError:
        raise Exception("Invalid response format.")


def fetch_geocode_by_address(
    api_key: str, address: str, timestamp: int, signature: str
) -> List[Dict[str, Any]]:
    """
    Fetch geocode data by address.

    Args:
        api_key (str): The API key for authentication.
        address (str): The address (e.g., "3305 Northland Dr, Austin, Tx").
        timestamp (int): The timestamp for the request.
        signature (str): The signature for the request.

    Returns:
        List[Dict[str, Any]]: Geocode data for the specified address.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    url = f"https://api.velocityweather.com/v1/{api_key}/reports/geocode/address.json"
    params = {"location": address, "ts": timestamp, "sig": signature}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("geocode", {}).get("data", [])
    except requests.RequestException as e:
        raise Exception(
            f"Failed to fetch geocode data by address: {e}"
        )
    except ValueError:
        raise Exception("Invalid response format.")


def fetch_geocode_by_zip(
    api_key: str,
    zip_code: str,
    us: int,
    timestamp: int,
    signature: str,
) -> List[Dict[str, Any]]:
    """
    Fetch geocode data by zip code.

    Args:
        api_key (str): The API key for authentication.
        zip_code (str): The zip code (e.g., "13060").
        us (int): Indicator for US zip code (1 for US, 0 for other).
        timestamp (int): The timestamp for the request.
        signature (str): The signature for the request.

    Returns:
        List[Dict[str, Any]]: Geocode data for the specified zip code.

    Raises:
        Exception: If the request fails or the response is invalid.
    """
    url = f"https://api.velocityweather.com/v1/{api_key}/reports/geocode/zip.json"
    params = {
        "zip": zip_code,
        "us": us,
        "ts": timestamp,
        "sig": signature,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("geocode", {}).get("data", [])
    except requests.RequestException as e:
        raise Exception(
            f"Failed to fetch geocode data by zip code: {e}"
        )
    except ValueError:
        raise Exception("Invalid response format.")
