import requests
import json
from datetime import date, datetime, timedelta
import os
from ..tool import Tool

from typing import Optional, Dict


def build_tool(config) -> Tool:
    tool = Tool(
        "Real Estate Information",
        "Look up rental and housing information",
        name_for_model="Zillow",
        description_for_model="Plugin for look up real estate information",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    BASE_URL = "https://zillow-com1.p.rapidapi.com"
    KEY = config["subscription_key"]
    HEADERS = {"X-RapidAPI-Key": KEY, "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"}

    @tool.get("/search_properties")
    def search_properties(
        location: str,
        page: Optional[int] = None,
        status_type: Optional[str] = None,
        home_type: Optional[str] = None,
        sort: Optional[str] = None,
        polygon: Optional[str] = None,
        minPrice: Optional[float] = None,
        maxPrice: Optional[float] = None,
        rentMinPrice: Optional[float] = None,
        rentMaxPrice: Optional[float] = None,
        bathsMin: Optional[int] = None,
        bathsMax: Optional[int] = None,
        bedsMin: Optional[int] = None,
        bedsMax: Optional[int] = None,
        sqftMin: Optional[int] = None,
        sqftMax: Optional[int] = None,
        buildYearMin: Optional[int] = None,
        buildYearMax: Optional[int] = None,
        daysOn: Optional[str] = None,
        soldInLast: Optional[str] = None,
        isBasementFinished: Optional[int] = None,
        isBasementUnfinished: Optional[int] = None,
        isPendingUnderContract: Optional[int] = None,
        isAcceptingBackupOffers: Optional[int] = None,
        isComingSoon: Optional[bool] = None,
        otherListings: Optional[int] = None,
        isNewConstruction: Optional[bool] = None,
        keywords: Optional[str] = None,
        lotSizeMin: Optional[str] = None,
        lotSizeMax: Optional[str] = None,
        saleByAgent: Optional[str] = None,
        saleByOwner: Optional[str] = None,
        isForSaleForeclosure: Optional[bool] = None,
        isWaterfront: Optional[bool] = None,
        hasPool: Optional[bool] = None,
        hasAirConditioning: Optional[bool] = None,
        isCityView: Optional[bool] = None,
        isMountainView: Optional[bool] = None,
        isWaterView: Optional[bool] = None,
        isParkView: Optional[bool] = None,
        isOpenHousesOnly: Optional[bool] = None,
        is3dHome: Optional[bool] = None,
        coordinates: Optional[str] = None,
        hoa: Optional[float] = None,
        includeHomesWithNoHoaData: Optional[bool] = None,
        isAuction: Optional[bool] = None,
    ):
        """
        Function to search properties based on a set of parameters.

        Parameters:
        location (str): Location details, address, county, neighborhood or Zip code.
        page (int): Page number if at the previous response totalPages > 1.
        status_type (str): Status type of the property.
        home_type (str): Type of the home.
        sort (str): Sorting option for the results.
        polygon (str): Polygon coordinates for the search.
        minPrice (float): Minimum price of the property.
        maxPrice (float): Maximum price of the property.
        rentMinPrice (float): Minimum rent price of the property.
        rentMaxPrice (float): Maximum rent price of the property.
        bathsMin (int): Minimum number of bathrooms.
        bathsMax (int): Maximum number of bathrooms.
        bedsMin (int): Minimum number of bedrooms.
        bedsMax (int): Maximum number of bedrooms.
        sqftMin (int): Minimum square feet area of the property.
        sqftMax (int): Maximum square feet area of the property.
         buildYearMin (int): Minimum year of construction of the property.
        buildYearMax (int): Maximum year of construction of the property.
        daysOn (str): Days on Zillow.
        soldInLast (str): Property sold in the last days.
        isBasementFinished (int): Whether the basement is finished or not.
        isBasementUnfinished (int): Whether the basement is unfinished or not.
        isPendingUnderContract (int): Whether the property is under contract or not.
        isAcceptingBackupOffers (int): Whether the property is accepting backup offers or not.
        isComingSoon (bool): Whether the property is coming soon or not.
        otherListings (int): Whether to include other listings or not.
        isNewConstruction (bool): Whether the property is new construction or not.
        keywords (str): Keywords to filter the search.
        lotSizeMin (str): Minimum lot size of the property.
        lotSizeMax (str): Maximum lot size of the property.
        saleByAgent (str): Whether the property is for sale by agent or not.
        saleByOwner (str): Whether the property is for sale by owner or not.
        isForSaleForeclosure (bool): Whether the property is for sale by foreclosure or not.
        isWaterfront (bool): Whether the property is a waterfront property or not.
        hasPool (bool): Whether the property has a pool or not.
        hasAirConditioning (bool): Whether the property has air conditioning or not.
        isCityView (bool): Whether the property has a city view or not.
        isMountainView (bool): Whether the property has a mountain view or not.
        isWaterView (bool): Whether the property has a water view or not.
        isParkView (bool): Whether the property has a park view or not.
        isOpenHousesOnly (bool): Whether to only include properties with open houses.
        is3dHome (bool): Whether the property has a 3D home tour.
        coordinates (str): Coordinates of the location for the search.
        hoa (float): Maximum HOA.
        includeHomesWithNoHoaData (bool): Whether to include homes with no HOA data.
        isAuction (bool): Whether the property is for auction.

        Returns:
        A response object from the Zillow API.
        """
        params = {
            "location": location,
            "page": page,
            "status_type": status_type,
            "home_type": home_type,
            "sort": sort,
            "polygon": polygon,
            "minPrice": minPrice,
            "maxPrice": maxPrice,
            "rentMinPrice": rentMinPrice,
            "rentMaxPrice": rentMaxPrice,
            "bathsMin": bathsMin,
            "bathsMax": bathsMax,
            "bedsMin": bedsMin,
            "bedsMax": bedsMax,
            "sqftMin": sqftMin,
            "sqftMax": sqftMax,
            "buildYearMin": buildYearMin,
            "buildYearMax": buildYearMax,
            "daysOn": daysOn,
            "soldInLast": soldInLast,
            "isBasementFinished": isBasementFinished,
            "isBasementUnfinished": isBasementUnfinished,
            "isPendingUnderContract": isPendingUnderContract,
            "isAcceptingBackupOffers": isAcceptingBackupOffers,
            "isComingSoon": isComingSoon,
            "otherListings": otherListings,
            "isNewConstruction": isNewConstruction,
            "keywords": keywords,
            "lotSizeMin": lotSizeMin,
            "lotSizeMax": lotSizeMax,
            "saleByAgent": saleByAgent,
            "saleByOwner": saleByOwner,
            "isForSaleForeclosure": isForSaleForeclosure,
            "isWaterfront": isWaterfront,
            "hasPool": hasPool,
            "hasAirConditioning": hasAirConditioning,
            "isCityView": isCityView,
            "isMountainView": isMountainView,
            "isWaterView": isWaterView,
            "isParkView": isParkView,
            "isOpenHousesOnly": isOpenHousesOnly,
            "is3dHome": is3dHome,
            "coordinates": coordinates,
            "hoa": hoa,
            "includeHomesWithNoHoaData": includeHomesWithNoHoaData,
            "isAuction": isAuction,
        }

        # Remove parameters that are None
        params = {k: v for k, v in params.items() if v is not None}
        url = BASE_URL + "/propertyExtendedSearch"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

        return response.json()

    @tool.get("/rent_estimate")
    def rent_estimate(
        propertyType: str,
        address: Optional[str] = None,
        long: Optional[float] = None,
        lat: Optional[float] = None,
        d: Optional[float] = None,
        beds: Optional[int] = None,
        baths: Optional[int] = None,
        sqftMin: Optional[int] = None,
        sqftMax: Optional[int] = None,
    ):
        """
        Estimate rent for a property.

        Args:
        propertyType (str): Type of the property. This is a required parameter. Options are 'SingleFamily', 'Condo', 'MultiFamily', 'Townhouse', 'Apartment'
        address (str, optional): Address of the property.
        long (float, optional): Longitude of the property.
        lat (float, optional): Latitude of the property.
        d (float, optional): Diameter in miles. The max and low values are 0.5 and 0.05 respectively. The default value is 0.5.
        beds (int, optional): Number of bedrooms in the property.
        baths (int, optional): Number of bathrooms in the property.
        sqftMin (int, optional): Minimum square footage of the property.
        sqftMax (int, optional): Maximum square footage of the property.

        Returns:
        A response object from the Zillow API with rent estimate and comparable rentals information.
        """
        params = {
            "propertyType": propertyType,
            "address": address,
            "long": long,
            "lat": lat,
            "d": d,
            "beds": beds,
            "baths": baths,
            "sqftMin": sqftMin,
            "sqftMax": sqftMax,
        }

        # Remove parameters that are None
        params = {k: v for k, v in params.items() if v is not None}
        url = BASE_URL + "/rentEstimate"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

        return response.json()

    @tool.get("/zillow_property")
    def zillow_property(zpid: Optional[int] = None, property_url: Optional[str] = None):
        """
        Fetch property details and Zestimate value.

        Args:
        zpid (int, optional): Unique ID that Zillow gives to each property.
        property_url (str, optional): Full page URL of the property on Zillow.

        Returns:
        A response object from the Zillow API with property details and Zestimate value.
        """
        params = {
            "zpid": zpid,
            "property_url": property_url,
        }

        # Remove parameters that are None
        params = {k: v for k, v in params.items() if v is not None}
        url = BASE_URL + "/property"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

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

    @tool.get("/property_by_mls")
    def property_by_mls(mls: str):
        """
        Search property by MLS #.

        Args:
        mls (str): MLS # of the property. This is a required parameter.

        Returns:
        A response object from the Zillow API with an array of zpid.
        """
        params = {
            "mls": mls,
        }
        url = BASE_URL + "/propertyByMls"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

        return response.json()

    @tool.get("/location_suggestions")
    def location_suggestions(q: str):
        """
        Search location by name.

        Args:
        q (str): Name of the state, county, neighborhood, city, or street. This is a required parameter.

        Returns:
        A response object from the Zillow API with suggested locations.
        """
        params = {
            "q": q,
        }
        url = BASE_URL + "/locationSuggestions"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

        return response.json()

    @tool.get("/similar_property")
    def similar_property(
        zpid: Optional[int] = None, property_url: Optional[str] = None
    ):
        """
        Get similar properties for sale. Either zpid or property_url is a required parameter.

        Args:
        zpid (int, optional): Zillow's unique identifier for a property. This can be obtained from /propertyExtendedSearch
            or /propertyByCoordinates endpoints, or extracted from a full URL.
        property_url (str, optional): Full page URL of the property.

        Returns:
        A response object from the Zillow API with similar properties for sale.
        """
        if not zpid and not property_url:
            raise ValueError("Either zpid or property_url must be provided.")

        params = {}
        if zpid:
            params["zpid"] = zpid
        if property_url:
            params["property_url"] = property_url
        url = BASE_URL + "/similarProperty"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

        return response.json()

    @tool.get("/similar_for_rent")
    def similar_for_rent(
        zpid: Optional[int] = None, property_url: Optional[str] = None
    ):
        """
        Get similar properties for rent. Either zpid or property_url is a required parameter.

        Args:
        zpid (int, optional): Zillow's unique identifier for a property. This can be obtained from /propertyExtendedSearch
            or /propertyByCoordinates endpoints, or extracted from a full URL.
        property_url (str, optional): Full page URL of the property.

        Returns:
        A response object from the Zillow API with similar properties for rent.
        """
        if not zpid and not property_url:
            raise ValueError("Either zpid or property_url must be provided.")

        params = {}
        if zpid:
            params["zpid"] = zpid
        if property_url:
            params["property_url"] = property_url
        url = BASE_URL + "/similarForRent"
        # Send GET request to Zillow API endpoint
        response = requests.get(url, headers=HEADERS, params=params)

        return response.json()

    return tool
