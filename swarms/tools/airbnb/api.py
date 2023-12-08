import requests
import json
from datetime import date, datetime, timedelta
import os
from ..tool import Tool

from typing import Optional, Dict, List


def build_tool(config) -> Tool:
    tool = Tool(
        "Short-term rental and housing information",
        "Look up rental and housing information",
        name_for_model="Airbnb",
        description_for_model="Plugin for look up rental and housing information",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    BASE_URL = "https://airbnb19.p.rapidapi.com/api/v1"
    KEY = config["subscription_key"]
    HEADERS = {"X-RapidAPI-Key": KEY, "X-RapidAPI-Host": "airbnb19.p.rapidapi.com"}

    @tool.get("/ssearch_property")
    def search_property(
        _id: str,
        display_name: Optional[str] = None,
        total_records: Optional[str] = "10",
        currency: Optional[str] = "USD",
        offset: Optional[str] = None,
        category: Optional[str] = None,
        adults: Optional[int] = 1,
        children: Optional[int] = None,
        infants: Optional[int] = None,
        pets: Optional[int] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        priceMin: Optional[int] = None,
        priceMax: Optional[int] = None,
        minBedrooms: Optional[int] = None,
        minBeds: Optional[int] = None,
        minBathrooms: Optional[int] = None,
        property_type: Optional[List[str]] = None,
        host_languages: Optional[List[str]] = None,
        amenities: Optional[List[str]] = None,
        type_of_place: Optional[List[str]] = None,
        top_tier_stays: Optional[List[str]] = None,
        self_check_in: Optional[bool] = None,
        instant_book: Optional[bool] = None,
        super_host: Optional[bool] = None,
        languageId: Optional[str] = None,
    ) -> dict:
        """
        This function takes various parameters to search properties on Airbnb.

        Parameters:
        api_key (str): The RapidAPI Key for Airbnb API.
        id (str): The ID of the destination.
        display_name (Optional[str]): The name of the destination.
        total_records (Optional[str]): The number of records to be retrieved per API call.
        currency (Optional[str]): The currency for the transaction.
        offset (Optional[str]): The offset for the search result.
        category (Optional[str]): The category of the properties.
        adults (Optional[int]): The number of adults.
        children (Optional[int]): The number of children.
        infants (Optional[int]): The number of infants.
        pets (Optional[int]): The number of pets.
        checkin (Optional[str]): The check-in date.
        checkout (Optional[str]): The check-out date.
        priceMin (Optional[int]): The minimum price.
        priceMax (Optional[int]): The maximum price.
        minBedrooms (Optional[int]): The minimum number of bedrooms.
        minBeds (Optional[int]): The minimum number of beds.
        minBathrooms (Optional[int]): The minimum number of bathrooms.
        property_type (Optional[List[str]]): The type of the property.
        host_languages (Optional[List[str]]): The languages that the host can speak.
        amenities (Optional[List[str]]): The amenities provided by the property.
        type_of_place (Optional[List[str]]): The type of the place.
        top_tier_stays (Optional[List[str]]): The list of top-tier stays.
        self_check_in (Optional[bool]): If the property has self check-in feature.
        instant_book (Optional[bool]): If the property can be booked instantly.
        super_host (Optional[bool]): If the host is a super host.
        languageId (Optional[str]): The ID of the language for the response.

        Returns:
        dict: A dictionary that contains the search results.
        """

        params = {
            "id": _id,
            "display_name": display_name,
            "totalRecords": total_records,
            "currency": currency,
            "offset": offset,
            "category": category,
            "adults": adults,
            "children": children,
            "infants": infants,
            "pets": pets,
            "checkin": checkin,
            "checkout": checkout,
            "priceMin": priceMin,
            "priceMax": priceMax,
            "minBedrooms": minBedrooms,
            "minBeds": minBeds,
            "minBathrooms": minBathrooms,
            "property_type": property_type,
            "host_languages": host_languages,
            "amenities": amenities,
            "type_of_place": type_of_place,
            "top_tier_stays": top_tier_stays,
            "self_check_in": self_check_in,
            "instant_book": instant_book,
            "super_host": super_host,
            "languageId": languageId,
        }
        response = requests.get(
            f"{BASE_URL}/searchPropertyByPlace", headers=HEADERS, params=params
        )
        return response.json()["data"][0]

    @tool.get("/search_property_by_coordinates")
    def search_property_by_coordinates(
        neLat: float,
        neLng: float,
        swLat: float,
        swLng: float,
        currency: Optional[str] = "USD",
        nextPageCursor: Optional[str] = None,
        totalRecords: Optional[str] = None,
        infants: Optional[int] = None,
        adults: Optional[int] = 1,
        children: Optional[int] = None,
        pets: Optional[int] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        priceMin: Optional[int] = None,
        priceMax: Optional[int] = None,
        minBedrooms: Optional[int] = None,
        minBeds: Optional[int] = None,
        minBathrooms: Optional[int] = None,
        property_type: Optional[List[str]] = None,
        host_languages: Optional[List[str]] = None,
        amenities: Optional[List[str]] = None,
        type_of_place: Optional[List[str]] = None,
        top_tier_stays: Optional[List[str]] = None,
        super_host: Optional[bool] = None,
    ) -> dict:
        """
        This function takes GEO coordinates and various other parameters to search properties on Airbnb.

        Parameters:
        neLat (float): Latitude of the northeastern corner of the search area.
        neLng (float): Longitude of the northeastern corner of the search area.
        swLat (float): Latitude of the southwestern corner of the search area.
        swLng (float): Longitude of the southwestern corner of the search area.
        Other parameters are the same as search_property function.

        Returns:
        dict: A dictionary that contains the search results.
        """

        params = {
            "neLat": neLat,
            "neLng": neLng,
            "swLat": swLat,
            "swLng": swLng,
            "currency": currency,
            "nextPageCursor": nextPageCursor,
            "totalRecords": totalRecords,
            "infants": infants,
            "adults": adults,
            "children": children,
            "pets": pets,
            "checkin": checkin,
            "checkout": checkout,
            "priceMin": priceMin,
            "priceMax": priceMax,
            "minBedrooms": minBedrooms,
            "minBeds": minBeds,
            "minBathrooms": minBathrooms,
            "property_type": property_type,
            "host_languages": host_languages,
            "amenities": amenities,
            "type_of_place": type_of_place,
            "top_tier_stays": top_tier_stays,
            "super_host": super_host,
        }
        response = requests.get(
            f"https://airbnb19.p.rapidapi.com/api/v2/searchPropertyByGEO",
            headers=HEADERS,
            params=params,
        )
        return response.json()["data"]["list"][0]

    @tool.get("/search_destination")
    def search_destination(self, query: str, country: Optional[str] = None) -> dict:
        """
        This function performs a destination search given a query and optionally a country. And return positions 'ID' information.

        Parameters:
        query (str): The search query.
        country (Optional[str]): The country for the search.

        Returns:
        dict: A dictionary that contains the search results. including ID information for a destination
        """

        params = {"query": query, "country": country}
        response = requests.get(
            f"{BASE_URL}/searchDestination", headers=HEADERS, params=params
        )
        return response.json()

    @tool.get("/property_by_coordinates")
    def property_by_coordinates(
        long: float,
        lat: float,
        d: Optional[float] = None,
        includeSold: Optional[bool] = None,
    ):
        """
        Search property by coordinates.

        Args:
        long (float): Longitude of the property. This is a required parameter.
        lat (float): Latitude of the property. This is a required parameter.
        d (float, optional): Diameter in miles. The max and low values are 0.5 and 0.05 respectively. The default value is 0.1.
        includeSold (bool, optional): Include sold properties in the results. True or 1 to include (default), False or 0 to exclude.

        Returns:
        A response object from the Zillow API with an array of zpid.
        """
        params = {
            "long": long,
            "lat": lat,
            "d": d,
            "includeSold": includeSold,
        }

        # Remove parameters that are None
        params = {k: v for k, v in params.items() if v is not None}
        url = BASE_URL + "/propertyByCoordinates"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

        return response.json()

    @tool.get("/get_property_details")
    def get_property_details(
        propertyId: int,
        currency: Optional[str] = "USD",
        checkIn: Optional[str] = None,
        checkOut: Optional[str] = None,
        adults: Optional[int] = 1,
        children: Optional[int] = None,
        infants: Optional[int] = None,
        pets: Optional[int] = None,
        languageId: Optional[str] = None,
    ) -> dict:
        """
        This function retrieves the details of a property given its ID.

        Parameters:
        propertyId (int): The ID of the property.
        currency (Optional[str]): The currency for the transaction.
        checkIn (Optional[str]): The check-in date.
        checkOut (Optional[str]): The check-out date.
        adults (Optional[int]): The number of adults.
        children (Optional[int]): The number of children.
        infants (Optional[int]): The number of infants.
        pets (Optional[int]): The number of pets.
        languageId (Optional[str]): The ID of the language for the response.

        Returns:
        dict: A dictionary that contains the details of the property.
        """

        params = {
            "propertyId": propertyId,
            "currency": currency,
            "checkIn": checkIn,
            "checkOut": checkOut,
            "adults": adults,
            "children": children,
            "infants": infants,
            "pets": pets,
            "languageId": languageId,
        }
        response = requests.get(
            f"https://airbnb19.p.rapidapi.com/api/v2/getPropertyDetails",
            headers=HEADERS,
            params=params,
        )
        return response.json()

    @tool.get("/check_availability")
    def check_availability(propertyId: int) -> dict:
        """
        This function checks the availability of a property given its ID.

        Parameters:
        propertyId (int): The ID of the property.

        Returns:
        dict: A dictionary that contains the availability of the property.
        """
        params = {
            "propertyId": propertyId,
        }
        response = requests.get(
            f"{BASE_URL}/checkAvailability", headers=HEADERS, params=params
        )
        return response.json()

    @tool.get("/get_property_reviews")
    def get_property_reviews(propertyId: int) -> dict:
        """
        This function retrieves the reviews of a property given its ID.

        Parameters:
        propertyId (int): The ID of the property.

        Returns:
        dict: A dictionary that contains the reviews of the property.
        """
        params = {
            "propertyId": propertyId,
        }
        response = requests.get(
            f"{BASE_URL}/getPropertyReviews", headers=HEADERS, params=params
        )
        return response.json()

    @tool.get("/get_property_checkout_price")
    def get_property_checkout_price(propertyId: int, checkIn: str) -> dict:
        """
        This function retrieves the checkout cost of a property given its ID and check-in date.

        Parameters:
        propertyId (int): The ID of the property.
        checkIn (str): The check-in date.

        Returns:
        dict: A dictionary that contains the checkout price of the property.
        """
        params = {"propertyId": propertyId, "checkIn": checkIn}
        response = requests.get(
            f"{BASE_URL}/getPropertyCheckoutPrice", headers=HEADERS, params=params
        )
        return response.json()

    return tool
