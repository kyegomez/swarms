from pydantic import BaseModel, Field


class RequestMetarNearest(BaseModel):
    latitude: str = Field(
        ...,
        description=(
            "The latitude of the location for which the nearest METAR"
            " station is requested."
        ),
    )
    longitude: str = Field(
        ...,
        description=(
            "The longitude of the location for which the nearest"
            " METAR station is requested."
        ),
    )


class PointQueryPrecipTotalAccum24Hr(BaseModel):
    layer: str = Field(
        ...,
        description=(
            "The layer of the precipitation total accumulation in the"
            " last 24 hours."
        ),
    )
    projection: str = Field(
        ...,
        description=(
            "The projection of the location for which the"
            " precipitation total accumulation is requested."
        ),
    )
    longitude: float = Field(
        ...,
        description=(
            "The longitude of the location for which the"
            " precipitation total accumulation is requested."
        ),
    )
    latitude: float = Field(
        ...,
        description=(
            "The latitude of the location for which the precipitation"
            " total accumulation is requested."
        ),
    )


class RequestNDFDBasic(BaseModel):
    latitude: float = Field(
        ...,
        description=(
            "The latitude of the location for which the NDFD basic"
            " forecast is requested."
        ),
    )
    longitude: float = Field(
        ...,
        description=(
            "The longitude of the location for which the NDFD basic"
            " forecast is requested."
        ),
    )
    forecast_time: str = Field(
        ...,
        description=(
            "The forecast time for which the NDFD basic forecast is"
            " requested."
        ),
    )


class PointQueryBaronHiresMaxReflectivityDbzAll(BaseModel):
    layer: str = Field(
        ...,
        description=(
            "The layer of the maximum reflectivity in dBZ for all"
            " heights."
        ),
    )
    projection: str = Field(
        ...,
        description=(
            "The projection of the location for which the maximum"
            " reflectivity is requested."
        ),
    )
    longitude: float = Field(
        ...,
        description=(
            "The longitude of the location for which the maximum"
            " reflectivity is requested."
        ),
    )
    latitude: float = Field(
        ...,
        description=(
            "The latitude of the location for which the maximum"
            " reflectivity is requested."
        ),
    )


class PointQueryBaronHiresWindSpeedMph10Meter(BaseModel):
    layer: str = Field(
        ...,
        description=(
            "The layer of the wind speed in mph at 10 meters above"
            " ground level."
        ),
    )
    projection: str = Field(
        ...,
        description=(
            "The projection of the location for which the wind speed"
            " is requested."
        ),
    )
    longitude: float = Field(
        ...,
        description=(
            "The longitude of the location for which the wind speed"
            " is requested."
        ),
    )
    latitude: float = Field(
        ...,
        description=(
            "The latitude of the location for which the wind speed is"
            " requested."
        ),
    )


def _remove_a_key(d: dict, remove_key: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key and "type" in d.keys():
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)
