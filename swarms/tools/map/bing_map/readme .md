# Map Service

Contributor: [Kunlun Zhu](https://github.com/Kunlun-Zhu)

# Map Info from Bing Map API

This Markdown document provides an introduction to the `Map Info` function implemented in the given code. The function allows users to look up map information using the Bing Map API. It provides functionality for retrieving distances between locations, routes between locations, coordinates of a location, and searching for nearby places.

## Function Summary

The `Map Info` function is a tool that enables users to look up map information using the Bing Map API. It provides features such as retrieving distances between locations, routes between locations, coordinates of a location, and searching for nearby places.

## Usage

To use the `Map Info` function, follow the steps below:

1. Set up the required configuration parameters, including the `subscription_key`.
2. Call the desired endpoint with appropriate parameters to perform the specific map-related operation.
3. The function returns the requested map information or relevant data.

## Endpoints

The `Map Info` tool provides the following endpoints:

### `/get_distance`

- **Method**: GET
- **Parameters**:
  - `start`: The starting location address or coordinates.
  - `end`: The destination location address or coordinates.
- **Returns**: The distance in miles between the specified start and end locations.

### `/get_route`

- **Method**: GET
- **Parameters**:
  - `start`: The starting location address or coordinates.
  - `end`: The destination location address or coordinates.
- **Returns**: A list of route instructions for the specified start and end locations.

### `/get_coordinates`

- **Method**: GET
- **Parameters**:
  - `location`: The location to retrieve coordinates for.
- **Returns**: The coordinates (latitude, longitude) of the specified location.

### `/search_nearby`

- **Method**: GET
- **Parameters**:
  - `search_term`: The keyword to search for nearby places (default: "restaurant").
  - `latitude`: The latitude coordinate of the location (default: 0.0).
  - `longitude`: The longitude coordinate of the location (default: 0.0).
  - `places`: The location to search nearby places for. If specified, latitude and longitude will be automatically retrieved (default: 'unknown').
  - `radius`: The search radius in meters (default: 5000).
- **Returns**: A list of nearby places based on the specified parameters.

## Error Handling

If any invalid options or exceptions occur during the execution of the function, appropriate error messages will be returned.
