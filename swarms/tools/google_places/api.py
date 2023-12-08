import requests
import json
from ..tool import Tool
import os

from typing import Any, Dict, List, Optional

import googlemaps


class GooglePlacesAPIWrapper:
    def __init__(self, subscription_key) -> None:
        self.gplaces_api_key: str = subscription_key
        self.google_map_client = googlemaps.Client(self.gplaces_api_key)
        self.top_k_results: Optional[int] = None

    def run(self, query: str) -> str:
        """Run Places search and get k number of places that exists that match."""
        search_results = self.google_map_client.places(query)["results"]
        num_to_return = len(search_results)

        places = []

        if num_to_return == 0:
            return "Google Places did not find any places that match the description"

        num_to_return = (
            num_to_return
            if self.top_k_results is None
            else min(num_to_return, self.top_k_results)
        )

        for i in range(num_to_return):
            result = search_results[i]
            details = self.fetch_place_details(result["place_id"])

            if details is not None:
                places.append(details)

        return "\n".join([f"{i+1}. {item}" for i, item in enumerate(places)])

    def fetch_place_details(self, place_id: str) -> Optional[str]:
        try:
            place_details = self.google_map_client.place(place_id)
            formatted_details = self.format_place_details(place_details)
            return formatted_details
        except Exception as e:
            logging.error(f"An Error occurred while fetching place details: {e}")
            return None

    def format_place_details(self, place_details: Dict[str, Any]) -> Optional[str]:
        try:
            name = place_details.get("result", {}).get("name", "Unkown")
            address = place_details.get("result", {}).get(
                "formatted_address", "Unknown"
            )
            phone_number = place_details.get("result", {}).get(
                "formatted_phone_number", "Unknown"
            )
            website = place_details.get("result", {}).get("website", "Unknown")

            formatted_details = (
                f"{name}\nAddress: {address}\n"
                f"Phone: {phone_number}\nWebsite: {website}\n\n"
            )
            return formatted_details
        except Exception as e:
            logging.error(f"An error occurred while formatting place details: {e}")
            return None


def build_tool(config) -> Tool:
    tool = Tool(
        "google_places",
        "Look up for information about places and locations",
        name_for_model="google_places",
        description_for_model=(
            "A tool that query the Google Places API. "
            "Useful for when you need to validate or "
            "discover addressed from ambiguous text. "
            "Input should be a search query."
        ),
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )
    api_wrapper = GooglePlacesAPIWrapper(config["subscription_key"])

    @tool.get("/search_places")
    def search_places(query: str):
        """Run Places search."""
        return str(api_wrapper.run(query))

    return tool
