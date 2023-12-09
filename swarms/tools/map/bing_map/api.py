import requests
import os
import json
from ...tool import Tool


def build_tool(config) -> Tool:
    tool = Tool(
        "Map Info form bing map api",
        "Look up map information",
        name_for_model="BingMap",
        description_for_model="Plugin for look up map information",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    KEY = config["subscription_key"]
    BASE_URL = "http://dev.virtualearth.net/REST/V1/"

    @tool.get("/get_distance")
    def get_distance(start: str, end: str):
        """Get the distance between two locations in miles"""
        # Request URL
        url = (
            BASE_URL
            + "Routes/Driving?o=json&wp.0="
            + start
            + "&wp.1="
            + end
            + "&key="
            + KEY
        )
        # GET request
        r = requests.get(url)
        data = json.loads(r.text)
        # Extract route information
        route = data["resourceSets"][0]["resources"][0]
        # Extract distance in miles
        distance = route["travelDistance"]
        return distance

    @tool.get("/get_route")
    def get_route(start: str, end: str):
        """Get the route between two locations in miles"""
        # Request URL
        url = (
            BASE_URL
            + "Routes/Driving?o=json&wp.0="
            + start
            + "&wp.1="
            + end
            + "&key="
            + KEY
        )
        # GET request
        r = requests.get(url)
        data = json.loads(r.text)
        # Extract route information
        route = data["resourceSets"][0]["resources"][0]
        itinerary = route["routeLegs"][0]["itineraryItems"]
        # Extract route text information
        route_text = []
        for item in itinerary:
            if "instruction" in item:
                route_text.append(item["instruction"]["text"])
        return route_text

    @tool.get("/get_coordinates")
    def get_coordinates(location: str):
        """Get the coordinates of a location"""
        url = BASE_URL + "Locations"
        params = {"query": location, "key": KEY}
        response = requests.get(url, params=params)
        json_data = response.json()
        coordinates = json_data["resourceSets"][0]["resources"][0]["point"][
            "coordinates"
        ]
        return coordinates

    @tool.get("/search_nearby")
    def search_nearyby(
        search_term: str = "restaurant",
        latitude: float = 0.0,
        longitude: float = 0.0,
        places: str = "unknown",
        radius: int = 5000,
    ):  #  radius in meters
        """Search for places nearby a location, within a given radius, and return the results into a list"""
        url = BASE_URL + "LocalSearch"
        if places != "unknown":
            latitude = get_coordinates(places)[0]
            longitude = get_coordinates(places)[1]
        # Build the request query string
        params = {
            "query": search_term,
            "userLocation": f"{latitude},{longitude}",
            "radius": radius,
            "key": KEY,
        }

        # Make the request
        response = requests.get(url, params=params)

        # Parse the response
        response_data = json.loads(response.content)

        # Get the results
        results = response_data["resourceSets"][0]["resources"]
        addresses = []
        for result in results:
            name = result["name"]
            address = result["Address"]["formattedAddress"]
            addresses.append(f"{name}: {address}")
        return addresses

    return tool
