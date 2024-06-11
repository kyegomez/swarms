from weather_swarm.tools.tools import request_metar_nearest
from swarms import tool


@tool(
    name="RequestMetarNearest",
    description=(
        "Requests the nearest METAR (Meteorological Aerodrome Report)"
        " data based on the given latitude and longitude."
    ),
    return_string=False,
    return_dict=False,
)
def request_metar_nearest_new(lat: float, lon: float):
    """
    Requests the nearest METAR (Meteorological Aerodrome Report) data based on the given latitude and longitude.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.

    Returns:
        The METAR data for the nearest location.
    """
    return request_metar_nearest(lat, lon)


out = request_metar_nearest_new(37.7749, -122.4194)
print(out)
print(type(out))
