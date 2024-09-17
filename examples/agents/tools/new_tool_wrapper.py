from swarms import tool


# Create the wrapper to wrap the function
@tool(
    name="Geo Coordinates Locator",
    description=(
        "Locates geo coordinates with a city and or zip code"
    ),
    return_string=False,
    return_dict=False,
)
def send_api_request_to_get_geo_coordinates(
    city: str = None, zip: int = None
):
    return "Test"


# Run the function to get the schema
out = send_api_request_to_get_geo_coordinates()
print(out)
