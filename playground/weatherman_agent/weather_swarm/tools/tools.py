# coding: utf-8

import base64
import hashlib
import hmac
import shutil
import time
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
import os
import json
import codecs
from dotenv import load_dotenv
import datetime

from typeguard import typechecked
from typing import Union

load_dotenv()

latin1 = codecs.lookup("latin-1")

host = os.environ.get(
    "BARON_API_HOST", "http://api.velocityweather.com/v1"
)
access_key = os.environ.get("BARON_ACCESS_KEY", "Y5lHXZfgce7P")
access_key_secret = os.environ.get(
    "BARON_ACCESS_KEY_SECRET",
    "rcscpInzyLuweENUjUtFDmqLkK1N0EPeaWQRjy7er1",
)


@typechecked
def a2w(a: bytes) -> str:
    """
    Decodes a byte string using Latin-1 encoding and returns the first character of the decoded string.

    Args:
        a (bytes): The byte string to be decoded.

    Returns:
        str: The first character of the decoded string.
    """
    return latin1.decode(a)[0]


@typechecked
def sig(key: str, secret: str) -> str:
    """
    Generates a signed string using HMAC-SHA1 and base64 encoding.

    Args:
        key (str): The key used for signing.
        secret (str): The secret used for signing.

    Returns:
        str: The signed string in the format "sig={signature}&ts={timestamp}".
    """

    ts = "{:.0f}".format(time.time())
    to_sign = key + ":" + ts
    hashval = hmac.new(
        secret.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha1
    )
    sig = a2w(
        base64.urlsafe_b64encode(hashval.digest()).replace(
            b"=", b"%3D"
        )
    )
    return "sig={}&ts={}".format(sig, ts)


@typechecked
def sign_request(url: str, key: str, secret: str) -> str:
    """
    Returns a signed URL by appending the signature and timestamp.

    Args:
        url (str): The URL to be signed.
        key (str): The key used for signing.
        secret (str): The secret used for signing.

    Returns:
        str: The signed URL with the signature and timestamp appended as query parameters.
    """

    """Returns signed url"""

    signature = sig(key, secret)
    q = "?" if url.find("?") == -1 else "&"
    url += "{}{}".format(q, signature)
    return url


########## [START] API REQUESTS ##########
@typechecked
def request_pointquery_nws_watches_warning_all() -> str:
    """
    Constructs a URL for querying all NWS watches and warnings for a specific point and signs the request.

    Returns:
        str: The signed URL for the point query.
    """

    uri = "/reports/alert/all-poly/point.json?lat=29.70&lon=-80.41"
    url = "%s/%s%s" % (host, access_key, uri)
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_lightning_count() -> str:
    """
    Constructs a URL for querying the count of lightning strikes in a specified region and signs the request.

    Returns:
        str: The signed URL for the lightning count query.
    """

    uri = "/reports/lightning/count/region.json?w_lon=-160&e_lon=0&n_lat=-2&s_lat=-70"
    url = "%s/%s%s" % (host, access_key, uri)
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_storm_vector(sitecode: str) -> str:
    """
    Constructs a URL for querying the storm vector for a specific site and signs the request.

    Args:
        sitecode (str): The code of the site for which the storm vector is being queried.

    Returns:
        str: The signed URL for the storm vector query.
    """

    uri = "/reports/stormvector/station/%s.json" % (sitecode)
    url = "%s/%s%s" % (host, access_key, uri)
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_geocodeip() -> str:
    """
    Constructs a URL for querying the geocode information of an IP address and signs the request.

    Returns:
        str: The signed URL for the geocode IP query.
    """

    uri = "/reports/geocode/ipaddress.json"
    url = "%s/%s%s" % (host, access_key, uri)
    url = sign_request(url, access_key, access_key_secret)
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_forecast(lat: float, lon: float) -> dict:
    """
    Constructs a URL for querying a 7-day point forecast for a specific latitude and longitude, signs the request, and retrieves the forecast data.

    Args:
        lat (float): The latitude for the forecast query.
        lon (float): The longitude for the forecast query.

    Returns:
        dict: The forecast data for the specified point if the request is successful, otherwise an empty dictionary.
    """
    uri = "/reports/pointforecast/basic.json?days=7&lat={}&lon={}".format(lat, lon)
    url = "%s/%s%s" % (host, access_key, uri)
    url = sign_request(url, access_key, access_key_secret)

    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return {}
    except ValueError as e:
        print(e)
        return {}

    assert response.code == 200
    data = json.loads(response.read())

    forecast_data = data.get("pointforecast_basic", {}).get("data", {})
    if isinstance(forecast_data, dict):
        return forecast_data
    else:
        return {"forecast_data": forecast_data}


@typechecked
def request_metar_northamerica() -> None:
    """
    Constructs a URL for querying METAR data for North America, signs the request, and retrieves the data.
    Processes the METAR data and associated forecasts, then saves the data to a JSON file.

    Returns:
        None
    """

    uri = "/reports/metar/region.json?n_lat=51.618017&s_lat=23.241346&w_lon=-129.375000&e_lon=-60.644531"
    url = "%s/%s%s" % (host, access_key, uri)
    url = sign_request(url, access_key, access_key_secret)

    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return

    assert response.code == 200
    data = json.loads(response.read())

    metars = {}
    pages = data["metars"]["meta"]["pages"]

    print("processing {} pages of METAR data".format(pages))

    for i in range(1, pages + 1):
        print("processing page {}".format(i))
        page_url = url + "&page={}".format(i)
        try:
            response = urlopen(page_url)
        except URLError as e:
            print(e)
            return
        except ValueError as e:
            print(e)
            return

        assert response.code == 200
        data = json.loads(response.read())
        for metar in data["metars"]["data"]:
            siteid = metar["station"]["id"]
            print("processing site {}".format(siteid))
            forecast = request_forecast(
                metar["station"]["coordinates"][1],
                metar["station"]["coordinates"][0],
            )

            metars[siteid] = {"metar": metar, "forecast": forecast}

    with open("metar.json", "w") as metar_jsonfile:
        json.dump(metars, metar_jsonfile, indent=4, sort_keys=True)



@typechecked
def request_metar_nearest(lat: str, lon: str):
    """
    Requests the nearest METAR (Meteorological Aerodrome Report) data based on the given latitude and longitude.

    Args:
        lat (str): The latitude of the location.
        lon (str): The longitude of the location.

    Returns:
        str: The signed request URL for retrieving the METAR data.
    """
    uri = (
        "/reports/metar/nearest.json?lat=%s&lon=%s&within_radius=500&max_age=75"
        % (
            lat,
            lon,
        )
    )
    url = "%s/%s%s" % (host, access_key, uri)
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_metar(station_id: str) -> str:
    """
    Constructs a URL for querying METAR data for a specific station and signs the request.

    Args:
        station_id (str): The ID of the station for which the METAR data is being queried.

    Returns:
        str: The signed URL for the METAR query.
    """

    uri = "/reports/metar/station/%s.json" % station_id
    url = "%s/%s%s" % (host, access_key, uri)
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_ndfd_hourly(lat: float, lon: float, utc_datetime: datetime.datetime) -> str:
    """
    Requests NDFD hourly data for a specific latitude, longitude, and UTC datetime.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.
        utc_datetime (datetime.datetime): The UTC datetime for the request.

    Returns:
        str: The signed URL for the request.
    """
    datetime_str = (
        utc_datetime.replace(microsecond=0).isoformat() + "Z"
    )
    uri = f"/reports/ndfd/hourly.json?lat={lat}&lon={lon}&utc={datetime_str}"
    url = f"{host}/{access_key}{uri}"
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_ndfd_basic(lat: float, lon: float, utc_datetime: datetime.datetime) -> str:
    """
    Requests NDFD basic data for a specific latitude, longitude, and UTC datetime.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.
        utc_datetime (datetime.datetime): The UTC datetime for the request.

    Returns:
        str: The signed URL for the request.
    """

    datetime_str = (
        utc_datetime.replace(microsecond=0).isoformat() + "Z"
    )
    uri = f"/reports/ndfd/basic.json?lat={lat}&lon={lon}&utc={datetime_str}&days=7"
    url = f"{host}/{access_key}{uri}"
    return sign_request(url, access_key, access_key_secret)


@typechecked
def request_tile(product: str, product_config: str, z: int, x: int, y: int) -> None:
    """
    Requests a tile for a specific product and configuration, retrieves the data, and saves it as a PNG file.

    Args:
        product (str): The product name.
        product_config (str): The product configuration.
        z (int): The zoom level.
        x (int): The tile's x coordinate.
        y (int): The tile's y coordinate.

    Returns:
        None
    """

    url = "%s/%s/meta/tiles/product-instances/%s/%s" % (
        host,
        access_key,
        product,
        product_config,
    )
    url = sign_request(url, access_key, access_key_secret)

    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200
    data = json.loads(response.read())

    # Select the most recent product instance for this example.
    product_instance = data[0]

    url = "%s/%s/tms/1.0.0/%s+%s+%s/%d/%d/%d.png" % (
        host,
        access_key,
        product,
        product_config,
        product_instance["time"],
        z,
        x,
        y,
    )

    try:
        # If it's a forecast product, it will have valid_times. The latest one is used for this example.
        url += "?valid_time={}".format(
            product_instance["valid_times"][0]
        )
    except KeyError:
        pass

    url = sign_request(url, access_key, access_key_secret)
    print(url)
    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    print("headers:")
    print(
        json.dumps(
            response.headers._headers, indent=4, sort_keys=True
        )
    )

    content = response.read()
    filename = "./tms_img_{}_{}.png".format(product, product_config)
    print(
        "Read {} bytes, saving as {}".format(len(content), filename)
    )
    with open(filename, "wb") as f:
        f.write(content)


@typechecked
def point_query(product: str, product_config: str, lon: float, lat: float) -> None:
    """
    Queries the most recent 'time' and, if applicable, 'valid_time' for a given product and product configuration at a specified longitude and latitude point.

    Args:
        product (str): The product name.
        product_config (str): The product configuration.
        lon (float): The longitude of the location.
        lat (float): The latitude of the location.

    Returns:
        None
    """
    # Get the list of product instances.
    url = "{host}/{key}/meta/tiles/product-instances/{product}/{product_config}".format(
        host=host,
        key=access_key,
        product=product,
        product_config=product_config,
    )
    url = sign_request(url, access_key, access_key_secret)
    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200
    data = json.loads(response.read())

    # Select the most recent product instance for this example.
    product_instance = data[0]

    # Query our lon, lat point.
    url = "{host}/{key}/point/{product}/{product_config}/{product_instance}.{file_type}?lon={lon}&lat={lat}".format(
        host=host,
        key=access_key,
        product=product,
        product_config=product_config,
        product_instance=product_instance["time"],
        file_type="json",
        lon=lon,
        lat=lat,
    )

    try:
        if product_instance["valid_times"][0]:
            # If it's a forecast product, it will have valid_times. Display them all
            url += "&valid_time=*"

        # If it's a forecast product, it will have valid_times. The latest one is used for this example.
        # url += '&valid_time={}'.format(product_instance['valid_times'][0])

    except KeyError:
        pass

    url = sign_request(url, access_key, access_key_secret)
    print(url)

    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    content = response.read()
    charset = response.headers.get_param("charset")
    if charset:
        content = content.decode(charset)
    content = json.loads(content)

    print("headers:")
    print(
        json.dumps(
            response.headers._headers, indent=4, sort_keys=True
        )
    )
    print("content:")
    print(
        json.dumps(
            content, indent=4, sort_keys=True, ensure_ascii=False
        )
    )


@typechecked
def point_query_multi(product: str, product_config: str, points: 'list[tuple[float, float]]') -> None:
    """
    For the given product and product_config, queries the most recent 'time'
    (and most recent 'valid_time' if it's a forecast product) for a list of points.

    Args:
        product (str): The product name.
        product_config (str): The product configuration.
        points (list[tuple[float, float]]): A list of tuples, each containing the longitude and latitude of a point.

    Returns:
        None
    """

    """
    For the given product and product_config, queries the most recent 'time'
    (and most recent 'valid_time' if it's a forecast product).
    """

    # Get the list of product instances.
    url = "{host}/{key}/meta/tiles/product-instances/{product}/{product_config}".format(
        host=host,
        key=access_key,
        product=product,
        product_config=product_config,
    )
    url = sign_request(url, access_key, access_key_secret)
    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200
    data = json.loads(response.read())

    # Select the most recent product instance for this example.
    product_instance = data[0]

    def format_point(_p, _decimals=3):
        return ",".join(str(round(_, _decimals)) for _ in _p)

    # Query our list of lon, lat points
    url = "{host}/{key}/point/multi/{product}/{product_config}/{product_instance}.{file_type}?points={points}".format(
        host=host,
        key=access_key,
        product=product,
        product_config=product_config,
        product_instance=product_instance["time"],
        file_type="json",
        points="|".join(format_point(_) for _ in points),
    )

    try:
        # If it's a forecast product, it will have valid_times. The latest one is used for this example.
        url += "&valid_time={}".format(
            product_instance["valid_times"][0]
        )
    except KeyError:
        pass

    url = sign_request(url, access_key, access_key_secret)
    print(url)

    try:
        request = Request(url, headers={"Accept-Encoding": "gzip"})
        response = urlopen(request)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    if response.headers.get("Content-Encoding") == "gzip":
        import gzip
        import io

        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file, mode="rb")
        content = decompressed_file.read()
    else:
        content = response.read()

    charset = response.headers.get_param("charset")
    if charset:
        content = content.decode(charset)
    content = json.loads(content)

    print("headers:")
    print(
        json.dumps(
            response.headers._headers, indent=4, sort_keys=True
        )
    )
    print("content:")
    print(
        json.dumps(
            content, indent=4, sort_keys=True, ensure_ascii=False
        )
    )


@typechecked
def point_query_region(product: str, product_config: str, n_lat: float, s_lat: float, w_lon: float, e_lon: float) -> None:
    """
    For the given product and product_config, queries the most recent 'time'
    (and most recent 'valid_time' if it's a forecast product) for a specific region.

    Args:
        product (str): The product name.
        product_config (str): The product configuration.
        n_lat (float): The northern latitude of the region.
        s_lat (float): The southern latitude of the region.
        w_lon (float): The western longitude of the region.
        e_lon (float): The eastern longitude of the region.

    Returns:
        None
    """

    """
    For the given product and product_config, queries the most recent 'time'
    (and most recent 'valid_time' if it's a forecast product).
    """

    # Get the list of product instances.
    url = "{host}/{key}/meta/tiles/product-instances/{product}/{product_config}".format(
        host=host,
        key=access_key,
        product=product,
        product_config=product_config,
    )

    url = sign_request(url, access_key, access_key_secret)
    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200
    data = json.loads(response.read())

    # Select the most recent product instance for this example.
    product_instance = data[0]

    def format_value(_, _decimals=3):
        return str(round(_, _decimals))

    # Query our region
    url = "{host}/{key}/point/region/{product}/{product_config}/{product_instance}.{file_type}?n_lat={n_lat}&s_lat={s_lat}&w_lon={w_lon}&e_lon={e_lon}".format(
        host=host,
        key=access_key,
        product=product,
        product_config=product_config,
        product_instance=product_instance["time"],
        file_type="json",
        n_lat=format_value(n_lat),
        s_lat=format_value(s_lat),
        w_lon=format_value(w_lon),
        e_lon=format_value(e_lon),
    )

    try:
        # If it's a forecast product, it will have valid_times. The latest one is used for this example.
        url += "&valid_time={}".format(
            product_instance["valid_times"][0]
        )
    except KeyError:
        pass

    url = sign_request(url, access_key, access_key_secret)
    print(url)

    try:
        request = Request(url, headers={"Accept-Encoding": "gzip"})
        response = urlopen(request)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    if response.headers.get("Content-Encoding") == "gzip":
        import gzip
        import io

        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file, mode="rb")
        content = decompressed_file.read()
    else:
        content = response.read()

    charset = response.headers.get_param("charset")
    if charset:
        content = content.decode(charset)
    content = json.loads(content)

    print("headers:")
    print(
        json.dumps(
            response.headers._headers, indent=4, sort_keys=True
        )
    )
    print("content:")
    print(
        json.dumps(
            content, indent=4, sort_keys=True, ensure_ascii=False
        )
    )


@typechecked
def request_wms_capabilities(product: str, product_config: str) -> None:
    """
    Requests WMS capabilities for a specific product and product configuration, signs the request, and prints the response content.

    Args:
        product (str): The product name.
        product_config (str): The product configuration.

    Returns:
        None
    """

    url = "{}/{}/wms/{}/{}?VERSION=1.3.0&SERVICE=WMS&REQUEST=GetCapabilities".format(
        host, access_key, product, product_config
    )
    url = sign_request(url, access_key, access_key_secret)
    print(url)

    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    content = response.read()
    print(content)


@typechecked
def request_wms(product: str, product_config: str, image_size_in_pixels: 'list[int]', image_bounds: 'list[float]') -> None:
    """
    Requests a WMS image and saves it to disk in the current directory.

    Args:
        product (str): The product code, such as 'C39-0x0302-0'.
        product_config (str): The product configuration, such as 'Standard-Mercator' or 'Standard-Geodetic'.
        image_size_in_pixels (list[int]): The image width and height in pixels, such as [1024, 1024].
        image_bounds (list[float]): The bounds of the image. See below for details depending on the projection.

    A. If requesting a Mercator (EPSG:3857) image:
        1. The coordinates must be in meters.
        2. The WMS 1.3.0 spec requires the coordinates be in this order [xmin, ymin, xmax, ymax].
        3. As an example, to request the whole world, you would use [-20037508.342789244, -20037508.342789244, 20037508.342789244, 20037508.342789244].
           Because this projection stretches to infinity as you approach the poles, the ymin and ymax values
           are clipped to the equivalent of -85.05112877980659 and 85.05112877980659 latitude, not -90 and 90 latitude,
           resulting in a perfect square of projected meters.
    B. If requesting a Geodetic (EPSG:4326) image:
        1. The coordinates must be in decimal degrees.
        2. The WMS 1.3.0 spec requires the coordinates be in this order [lat_min, lon_min, lat_max, lon_max].
        3. As an example, to request the whole world, you would use [-90, -180, 90, 180].

    Theoretically it is possible to request any arbitrary combination of image_size_in_pixels and image_bounds,
    but this is not advisable and is actually discouraged. It is expected that the proportion you use for
    image_width_in_pixels/image_height_in_pixels is equal to image_width_bounds/image_height_bounds. If this is
    not the case, you have most likely done some incorrect calculations. It will result in a distorted (stretched
    or squished) image that is incorrect for the requested projection. One fairly obvious sign that your
    proportions don't match up correctly is that the image you receive from your WMS request will have no
    smoothing (interpolation), resulting in jaggy or pixelated data.

    Returns:
        None
    """
    # Convert the image bounds to a comma-separated string.
    image_bounds_str = ",".join(str(x) for x in image_bounds)

    # We're using the TMS-style product instances API here for simplicity. If you
    # are using a standards-compliant WMS client, do note that we also provide a
    # WMS-style API to retrieve product instances which may be more suitable to your
    # needs. See our documentation for details.

    # For this example, we use the optional parameter "page_size" to limit the
    # list of product instances to the most recent instance.
    meta_url = (
        "{}/{}/meta/tiles/product-instances/{}/{}?page_size=1".format(
            host, access_key, product, product_config
        )
    )
    meta_url = sign_request(meta_url, access_key, access_key_secret)

    try:
        response = urlopen(meta_url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    # Decode the product instance response and get the most recent product instance time,
    # to be used in the WMS image request.
    content = json.loads(response.read())
    product_instance = content[0]

    # WMS uses EPSG codes, while our product configuration code uses 'Geodetic' or
    # 'Mercator'. We map between the two here to prepare for the WMS CRS query parameter.
    epsg_code = (
        "EPSG:4326"
        if product_config.endswith("-Geodetic")
        else "EPSG:3857"
    )

    wms_url = "{}/{}/wms/{}/{}?VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap&CRS={}&LAYERS={}&BBOX={}&WIDTH={}&HEIGHT={}".format(
        host,
        access_key,
        product,
        product_config,
        epsg_code,
        product_instance["time"],
        image_bounds_str,
        image_size_in_pixels[0],
        image_size_in_pixels[1],
    )

    try:
        # If it's a forecast product, it will have valid_times. The latest one is used for this example.
        wms_url += "&TIME={}".format(
            product_instance["valid_times"][0]
        )
    except KeyError:
        pass

    wms_url = sign_request(wms_url, access_key, access_key_secret)
    print(wms_url)

    try:
        response = urlopen(wms_url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    content = response.read()
    filename = "./wms_img_{}_{}.png".format(product, product_config)
    print(
        "Read {} bytes, saving as {}".format(len(content), filename)
    )
    with open(filename, "wb") as f:
        f.write(content)



@typechecked
def request_geotiff(product: str, product_config: str, product_instance: str = "") -> 'tuple[str, dict]':
    """
    Requests a GeoTIFF image for a specific product, product configuration, and product instance.
    If no product instance is provided, the most recent instance is used.

    Args:
        product (str): The product code.
        product_config (str): The product configuration.
        product_instance (str, optional): The product instance time. Defaults to an empty string.

    Returns:
        tuple[str, dict]: The filename where the GeoTIFF is saved and a dictionary of valid times.
    """

    if not product_instance:
        # For this example, we use the optional parameter "page_size" to limit the
        # list of product instances to the most recent instance.
        meta_url = "{}/{}/meta/tiles/product-instances/{}/{}?page_size=1".format(
            host, access_key, product, product_config
        )
        meta_url = sign_request(
            meta_url, access_key, access_key_secret
        )

        try:
            response = urlopen(meta_url)
        except URLError as e:
            print(e)
            return
        except ValueError as e:
            print(e)
            return
        assert response.code == 200

        # Decode the product instance response and get the most recent product instance time,
        # to be used in the geotiff request.
        content = json.loads(response.read())
        product_instance = content[0]["time"]

    url = "/".join(
        [
            host,
            access_key,
            "geotiff",
            product,
            product_config,
            product_instance,
        ]
    )
    url = sign_request(url, access_key, access_key_secret)
    print(url)

    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    content = json.loads(response.read())
    url = content["source"]

    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200

    filename = "./{}.tif".format(
        "_".join([product, product_config, product_instance])
    )
    with open(filename, "wb") as f:
        # The geotiffs can be very large, so we don't want to read the
        # http body entirely into memory before writing -- copy it directly
        # to a file instead.
        shutil.copyfileobj(response, f)
    return filename, content.get("valid_times", {})


@typechecked
def bgfs_basic(lon: float, lat: float, date: Union[datetime.date, datetime.datetime], days: int = 1) -> None:
    """
    Requests BGFS basic data for a specific longitude, latitude, date, and number of days.
    
    Args:
        lon (float): The longitude of the location.
        lat (float): The latitude of the location.
        date (datetime.datetime): The date for the request.
        days (int, optional): The number of days for the request. Defaults to 1.

    Returns:
        None
    """

    url = "{host}/{key}/reports/bgfs/basic?lon={lon}&lat={lat}&utc={utc}&days={days}".format(
        host=host,
        key=access_key,
        lon=lon,
        lat=lat,
        utc=date.strftime("%Y-%m-%d"),
        days=days,
    )
    url = sign_request(url, access_key, access_key_secret)
    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200
    content = json.loads(response.read())

    # Convert back to json only so we can let the json library format the
    # response for pretty display.
    print(
        json.dumps(
            content, indent=4, sort_keys=True, ensure_ascii=False
        )
    )


@typechecked
def bgfs_extended(lon: float, lat: float, date: Union[datetime.date, datetime.datetime], days: int = 1) -> None:
    
    """
    Fetches extended weather reports using the BGFS API.

    Args:
        lon (float): The longitude of the location.
        lat (float): The latitude of the location.
        date (datetime.datetime): The date for which the weather reports are requested.
        days (int, optional): The number of days for which the weather reports are requested. Defaults to 1.

    Returns:
        None

    Raises:
        URLError: If there is an error in the URL request.
        ValueError: If there is an error in the URL parameters.
    """

    url = "{host}/{key}/reports/bgfs/extended?lon={lon}&lat={lat}&utc={utc}&days={days}".format(
        host=host,
        key=access_key,
        lon=lon,
        lat=lat,
        utc=date.strftime("%Y-%m-%d"),
        days=days,
    )
    url = sign_request(url, access_key, access_key_secret)
    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200
    content = json.loads(response.read())

    # Convert back to json only so we can let the json library format the
    # response for pretty display.
    print(
        json.dumps(
            content, indent=4, sort_keys=True, ensure_ascii=False
        )
    )


@typechecked
def bgfs_hourly(lon: float, lat: float, date_hour: Union[datetime.date, datetime.datetime], hours: int = 1) -> None:
    """
    Fetches hourly weather reports from the BGFS API for the given longitude, latitude, and date hour.

    Args:
        lon (float): The longitude of the location.
        lat (float): The latitude of the location.
        date_hour (datetime.datetime): The date and hour for which to fetch the weather reports.
        hours (int, optional): The number of hours of weather reports to fetch. Defaults to 1.

    Returns:
        None

    Raises:
        URLError: If there is an error in the URL request.
        ValueError: If there is an error in the URL parameters.
    """

    url = "{host}/{key}/reports/bgfs/hourly?lon={lon}&lat={lat}&utc={utc}&hours={hours}".format(
        host=host,
        key=access_key,
        lon=lon,
        lat=lat,
        utc=date_hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
        hours=hours,
    )
    url = sign_request(url, access_key, access_key_secret)
    try:
        response = urlopen(url)
    except URLError as e:
        print(e)
        return
    except ValueError as e:
        print(e)
        return
    assert response.code == 200
    content = json.loads(response.read())

    # Convert back to json only so we can let the json library format the
    # response for pretty display.
    print(
        json.dumps(
            content, indent=4, sort_keys=True, ensure_ascii=False
        )
    )


from typing import Iterator, Dict
@typechecked
def iter_product_instances(product: str, product_config: str, request_limit: int = 100) -> Iterator[Dict]:
    """
    Iterate over all available product instances, one by one, using a
    configurable number of instances per request.

    Args:
        product (str): The product code.
        product_config (str): The product configuration.
        request_limit (int, optional): The number of instances to request per API call. Defaults to 100.

    Yields:
        dict: A product instance.

    Returns:
        None
    """

    url_template = (
        "{}/{}/meta/tiles/product-instances/{}/{}?limit={}".format(
            host, access_key, product, product_config, request_limit
        )
    )
    url = url_template

    request_count = 0
    content_count = 0
    while content_count < request_limit:
        signed_url = sign_request(url, access_key, access_key_secret)
        request_count += 1
        try:
            response = urlopen(signed_url)
        except URLError as e:
            print(e)
            return
        except ValueError as e:
            print(e)
            return
        assert response.code == 200

        content = json.loads(response.read())
        for item in content:
            yield item

        content_count += len(content)

        if len(content) < request_limit:
            # We didn't get a full page, so we must be on the last page and
            # therefore -- finished.
            print(
                "Request count: {}. Instance count: {}.".format(
                    request_count,
                    (request_count - 1) * request_limit
                    + len(content),
                )
            )
            return
        url = "{}&older_than={}".format(
            url_template, content[-1]["time"]
        )


def test_api_calls():
    url = request_metar_nearest("38", "-96")
    print("*** request METAR nearest ***")
    print(url)
    print(urlopen(url).read())
    print("")

    point_query(
        "precip-totalaccum-24hr", "Standard-Mercator", -86.6, 34.4
    )

    forecast_time = datetime.datetime.utcnow()
    url = request_ndfd_basic(34.730301, -86.586098, forecast_time)
    print("*** request NDFD hourly ***")
    print(url)
    print(urlopen(url).read())
    print("")

    # /point/baron-hires-temp-f-2meter/Standard-Mercator/2024-05-02T12%3A00%3A00Z.jsonp?callback=_jqjsp&lat=30.173624550358536&lon=-95.3009033203125&ts=1714685100&sig=IOUh5xEZzyRqzT1MQctn1vxSqXM=&valid_time=*
    point_query(
        "baron-hires-maxreflectivity-dbz-all",
        "Mask1-Mercator",
        -86.6,
        34.4,
    )

    point_query(
        "baron-hires-windspeed-mph-10meter",
        "Standard-Mercator",
        -86.6,
        34.4,
    )

    # Get all product instances for a product.
    for i, instance in enumerate(iter_product_instances('C39-0x0302-0', 'Standard-Mercator')):
        print(type(instance))
        print('{:>3} {}'.format(i, instance['time']))

    # Or, alternatively, get the product instances using a wms-style request.
    request_wms_capabilities('C39-0x0302-0', 'Standard-Mercator')

    # Request the whole world in the EPSG:4326 projection. Note that the proportions for
    # the image size in pixels and the image bounds are identical (2:1).
    request_wms('C39-0x0302-0', 'Standard-Geodetic', [2048, 1024], [-90.0, -180.0, 90.0, 180.0])

    # Request the whole world in the EPSG:3857 projection. Note that the proportions for
    # the image size in pixels and the image bounds are identical (1:1).
    request_wms('C39-0x0302-0', 'Standard-Mercator', [2048, 2048], [-20037508.342789244, -20037508.342789244, 20037508.342789244, 20037508.342789244])

    filename, valid_times = request_geotiff('C39-0x0302-0', 'Standard-Mercator')

    
    print("*** request point query ***")
    point_query('C09-0x0331-0', 'Standard-Mercator', -86, 34)
    print("")


    # print("*** requesting METARS and Forecasts for North America ***")
    # request_metar_northamerica()
    # print("")


    url = request_metar("egll")
    print("*** request METAR ***")
    print(url)
    print(urlopen(url).read())
    print("")

    forecast_time = datetime.datetime.utcnow() + datetime.timedelta(hours=4)
    url = request_ndfd_hourly(34.730301, -86.586098, forecast_time)
    print("*** request NDFD hourly ***")
    print(url)
    print(urlopen(url).read())
    print("")

    request_tile("C39-0x0302-0", "Standard-Mercator", 1, 0, 1)
    url = request_storm_vector("mhx")
    print("*** request storm vectors ***")
    print(url)
    a = urlopen(url)
    print('JSON for storm vectors is %d bytes' % len(urlopen(url).read()))
    print("")
    url = request_geocodeip()
    print("*** geocode IP address ***")
    print(url)
    print(urlopen(url).read())
    print("")
    url = request_lightning_count()
    print("*** lightning count ***")
    print(url)
    print(urlopen(url).read())
    print("")

    date = datetime.datetime.now().date() + datetime.timedelta(days=1)
    bgfs_basic(-86.6, 34.4, date, 1)
    bgfs_extended(-86.6, 34.4, date, 1)
    bgfs_hourly(-86.6, 34.4, datetime.datetime.combine(date, datetime.time(hour=6)), 1)
    print("")

    point_query('C09-0x0331-0', 'Standard-Mercator', -86.6, 34.4)
    point_query_multi('C09-0x0331-0', 'Standard-Mercator', [(-86.6, 34.4), (-90.14, 38)])
    point_query_region('C09-0x0331-0', 'Standard-Mercator', 34.4, 34.1, -86.6, -86.5)

    


# if __name__ == "__main__":
#     main()
