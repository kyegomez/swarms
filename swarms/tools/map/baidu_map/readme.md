# Map Service

Contributor: [Kunlun Zhu](https://github.com/Kunlun-Zhu)

To Get Baidu API Key [BAIDU MAP API](https://lbsyun.baidu.com/apiconsole/key?application=key#/home)
After You Get Baidu API Key, chose 'SN校验方式' and obtain your SK from the setting page

# Map Info Tool

This Markdown document provides an introduction to the `Map Info` function implemented in the given code. The function allows users to look up map information in China using the Baidu Map API. It provides functionality for retrieving location coordinates, addresses by coordinates, nearby places, routes between locations, and distances between locations.

## Function Summary

The `Map Info` function is a tool that enables users to look up map information in China. It utilizes the Baidu Map API to provide various features related to locations, addresses, nearby places, routes, and distances.

## Usage

To use the `Map Info` function, follow the steps below:

1. Set up the required configuration parameters, including the `subscription_key` and `baidu_secret_key`.
2. Call the desired endpoint with appropriate parameters to perform the specific map-related operation.
3. The function returns the requested map information or relevant data.

## Endpoints

The `Map Info` tool provides the following endpoints:

### `/get_location`

- **Method**: GET
- **Parameters**:
  - `address`: The address to retrieve location coordinates for.
- **Returns**: The coordinates (latitude, longitude) of the specified address, or None if the coordinates cannot be obtained.

### `/get_address_by_coordinates`

- **Method**: GET
- **Parameters**:
  - `lat`: The latitude coordinate.
  - `lng`: The longitude coordinate.
- **Returns**: The location name corresponding to the given coordinates, or None if the address cannot be obtained.

### `/get_nearby_places`

- **Method**: GET
- **Parameters**:
  - `location`: The location coordinates (latitude, longitude) to search nearby places for.
  - `radius`: The search radius in meters.
  - `keyword`: The keyword to filter the search results (default: '餐厅' for restaurants).
- **Returns**: A list of nearby place names based on the specified location and search parameters.

### `/get_route`

- **Method**: GET
- **Parameters**:
  - `origin`: The origin address of the route.
  - `destination`: The destination address of the route.
- **Returns**: A list of route descriptions for the specified origin and destination, or None if the route cannot be obtained.

### `/get_distance`

- **Method**: GET
- **Parameters**:
  - `origin`: The origin address.
  - `destination`: The destination address.
- **Returns**: The distance in meters between the specified origin and destination.

## Error Handling

If any invalid options or exceptions occur during the execution of the function, appropriate error messages will be returned.
