import os
import re
import json
import requests
from ..tool import Tool
from typing import Optional
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from datetime import datetime, timedelta
from amadeus import Client, ResponseError


def build_tool(config) -> Tool:
    tool = Tool(
        "Travel Info",
        "Look up travel infomation about lodging, flight, car rental and landscape",
        name_for_model="Travel",
        description_for_model="""This is a plugin for look up real travel infomation. Results from this API are inaccessible for users. Please organize and re-present them.""",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
    if SERPAPI_KEY == "":
        raise RuntimeError(
            "SERPAPI_KEY not provided, please register one at https://serpapi.com/search-api and add it to environment variables."
        )

    AMADEUS_ID = os.environ.get("AMADEUS_ID", "")
    if AMADEUS_ID == "":
        raise RuntimeError(
            "AMADEUS_ID not provided, please register one following https://developers.amadeus.com/ and add it to environment variables."
        )

    AMADEUS_KEY = os.environ.get("AMADEUS_KEY", "")
    if AMADEUS_KEY == "":
        raise RuntimeError(
            "AMADEUS_KEY not provided, please register one following https://developers.amadeus.com/ and add it to environment variables."
        )

    def cName2coords(place_name: str, limits: Optional[int] = 1):
        """
        This function accepts a place name and returns its coordinates.
        :param (str) place_name: a string standing for target city for locating.
        :param (str) limits: number of searching results. Usually 1 is enough.
        :return: (longitude, latitude)
        """
        url = f"https://serpapi.com/locations.json?q={place_name}&limit={limits}"
        response = requests.get(url)
        if response.status_code == 200:
            if response.json():
                locations = response.json()
                return locations[0]["gps"]

            # if not a city, use google map to find this place
            else:
                params = {
                    "engine": "google_maps",
                    "q": place_name,
                    "type": "search",
                    "api_key": SERPAPI_KEY,
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                coords = results["place_results"]["gps_coordinates"]

                return coords["longitude"], coords["latitude"]

        else:
            return None

    def cName2IATA(city_name: str):
        """
        This function accepts a city name and returns a IATA pilot code of the local airport.
        :param (str) city_name: city name a s a string like 'Beijing'
        :return: 3-letter IATA code like 'PEK'

        """
        try:
            url = f"https://www.iata.org/en/publications/directories/code-search/?airport.search={city_name}"
            response = requests.get(url)
            html_content = response.content

            soup = str(BeautifulSoup(html_content, "html.parser"))
            head = soup.find(f"<td>{city_name}</td>")

            string = soup[head:]
            pattern = r"<td>(.*?)</td>"
            matches = re.findall(pattern, string)

            # Extract the desired value
            desired_value = matches[2]  # Index 2 corresponds to the third match (WUH)

            return desired_value  # Output: WUH

        except:
            raise ValueError("The input city may not have an IATA registered air-port.")

    @tool.get("/lodgingProducts")
    def lodgingProducts(
        destination: str,
        exhibit_maxnum: Optional[int] = 3,
    ):
        """
        This function returns the lodging resources near a given location.
        :param (str) destination: the destination can be a city, address, airport, or a landmark.
        :param (int, optional) exhibit_maxnum: int like 2, 3, 4. It determines how much items to exhibit.
        :return: lodging information at a given destination.
        """

        try:
            # Convert destination to geographic coordinates
            coords = cName2coords(destination)

            # Set parameters for Google search
            params = {
                "engine": "google_maps",
                "q": "hotel",
                "ll": f"@{coords[1]},{coords[0]},15.1z",
                "type": "search",
                "api_key": SERPAPI_KEY,
            }

            # Call GoogleSearch API with given parameters
            search = GoogleSearch(params)
            results = search.get_dict()
            local_results = results["local_results"]

            # hotel with website links are preferred
            filtered_results = sorted(
                local_results, key=lambda x: 1 if "website" in x else 0, reverse=True
            )[:exhibit_maxnum]

            # Return error message if no results match criteria
            if not filtered_results:
                return {"error": "No searching results satisfy user's demand."}

            # Parse and format relevant information for each result
            final_results = []
            for item in filtered_results:
                info = {}
                for metadata in (
                    "title",
                    "website",
                    "address",
                    "description",
                    "gps_coordinates",
                    "open_state",
                    "thumbnail",
                ):
                    if metadata in item:
                        info[metadata] = item[metadata]
                final_results.append(info)

            # Return formatted results along with recommendations for next steps
            return json.dumps({"data": final_results})

        except ResponseError as error:
            print("ResponseError", error)

    @tool.get("/flightProducts")
    def flightProducts(
        origin: str,
        destination: str,
        departureDate: Optional[str] = None,
        adult_num: Optional[str] = None,
        exhibit_maxnum: Optional[int] = 3,
    ):
        """
        This function returns the flight information between two cities.
        :param origin: (str) city name, origin of departure
        :param destination: (str) city name, destination of arrival
        :param departureDate: (str, optional) Date formatted as "yyyy-mm-dd". It shoule be LATER than the PRESENT date. Pass None if not sure about this.
        :param adult_num: (str, optional) Number of adults for flight tickets
        :param exhibit_maxnum: (int, optional) Maximum number of items to show
        :return: (dict) information about flight.

        """
        amadeus = Client(
            client_id=AMADEUS_ID,
            client_secret=AMADEUS_KEY,
        )

        # set default date if none or past date is given
        defaultDate = f"{(datetime.now() + timedelta(days=1)).year}-{'0' + str((datetime.now() + timedelta(days=1)).month) if (datetime.now() + timedelta(days=1)).month < 10 else (datetime.now() + timedelta(days=1)).month}-{(datetime.now() + timedelta(days=1)).day}"
        if not departureDate or departureDate < defaultDate:
            departureDate = defaultDate

        try:
            # Call API to search flight offers
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=cName2IATA(origin),
                destinationLocationCode=cName2IATA(destination),
                departureDate=departureDate,
                adults=adult_num,
            )

            # Filter results based on exhibit_maxnum
            filterd_results = response.data[:exhibit_maxnum]

            # If no results found return error message
            if not filterd_results:
                return {"error": "No search results satisfy user's demand."}

            final_results = []
            for item in filterd_results:
                info = {}
                metadata = [
                    "itineraries",
                    "travelerPricings",
                    "lastTicketingDate",
                    "numberOfBookableSeats",
                    "source",
                ]

                # Only include relevant metadata in info
                for key in metadata:
                    if key in item:
                        info[key] = item[key]

                final_results.append(info)

                # Return formatted results along with recommendations for next steps
                return json.dumps({"data": final_results})
        except ResponseError as error:
            print("ResponseError", error)

    @tool.get("/landscapeProducts")
    def landscapeProducts(
        destination: str,
        exhibit_maxnum: Optional[int] = 3,
    ):
        """
        This function returns the scenic spot information given a destination.
        :param (str) destination: string of cityname, destination of arrival
        :param (int, optional) exhibit_maxnum: int like 3, 4, 5. It determines how many spots to display.
        :return: Information about landscape around that destination.

        """

        try:
            # Get the coordinates of the destination using the cName2coords function
            coords = cName2coords(destination)

            # Set parameters for the GoogleSearch API call
            params = {
                "engine": "google_maps",
                "q": "tourist attractions",
                "ll": f"@{coords[1]},{coords[0]},15.1z",
                "type": "search",
                "api_key": SERPAPI_KEY,
            }

            # Call the GoogleSearch API
            search = GoogleSearch(params)
            results = search.get_dict()
            local_results = results["local_results"]

            # Sort the results by the specified keyword if provided
            sorting_keywords = "reviews"
            if sorting_keywords:
                local_results = sorted(
                    local_results,
                    key=lambda x: x[sorting_keywords] if sorting_keywords in x else 0,
                    reverse=True,
                )

            # Filter the results to exhibit_maxnum number of items
            filterd_results = local_results[:exhibit_maxnum]

            # Return an error message if no results are found
            if not filterd_results:
                return {"error": "No searching results satisfy user's demand."}

            # Format the results into a dictionary to be returned to the user
            final_results = []
            for item in filterd_results:
                final_results.append({"spot_name": item["title"]})

                for keywords in (
                    "description",
                    "address",
                    "website",
                    "rating",
                    "thumbnail",
                ):
                    if keywords in item:
                        final_results[-1][keywords] = item[keywords]

            # Return the formatted data and some extra information to the user
            return json.dumps({"data": final_results})

        except ResponseError as error:
            print("ResponseError", error)

    @tool.get("/carProducts")
    async def carProducts(
        pickup_location: str,
        exhibit_maxnum: Optional[int] = 3,
    ):
        """
        Given a pickup location, returns a list of car rentals nearby.
        :param pickup_location: string of city name or location for the car rental pickups.
        :param exhibit_maxnum: number of rental cars to display.
        :return: a dict of data and some extra-information for the LLM.
        """
        try:
            coords = cName2coords(pickup_location)
            # Construct search query params with SERPAPI_KEY
            params = {
                "engine": "google_maps",
                "q": "car rentals",
                "ll": f"@{coords[1]},{coords[0]},15.1z",
                "type": "search",
                "api_key": SERPAPI_KEY,
            }

            # Search for nearby car rentals on SERPAPI
            search = GoogleSearch(params)
            results = search.get_dict()
            local_results = results["local_results"]

            # Sort results by rating or reviews keywords if sorting_keywords is specified
            sorting_keywords = "reviews"
            if sorting_keywords:
                local_results = sorted(
                    local_results,
                    key=lambda x: x.get(sorting_keywords, 0),
                    reverse=True,
                )

            # Make sure spots with a website appear first
            local_results = sorted(
                local_results, key=lambda x: "website" in x, reverse=True
            )

            # Choose the exhibit_maxnum rentals to display
            filtered_results = local_results[:exhibit_maxnum]

            # Return an error if there are no results
            if not filtered_results:
                return {"error": "No results found."}

            # Format the output dictionary with relevant data
            final_results = []
            for item in filtered_results:
                spot = {"spot_name": item["title"]}

                for keyword in (
                    "description",
                    "address",
                    "website",
                    "rating",
                    "thumbnail",
                ):
                    if keyword in item:
                        spot[keyword] = item[keyword]

                final_results.append(spot)

                # Return the formatted output dictionary with extra information
                # Return formatted results along with recommendations for next steps
                return json.dumps({"data": final_results})

        except ResponseError:  # Handle response error exceptions
            return {"error": "Response error."}

    return tool
