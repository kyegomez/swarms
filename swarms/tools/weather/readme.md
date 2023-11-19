# Weather Queries

Contributor: [Yujia Qin](https://github.com/thuqinyj16)

You can get the API keys from https://www.weatherapi.com/

# Weather Info Tool

This tool allows you to look up weather information.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Weather Info"
- **description_for_model**: "Plugin for look up weather information"
- **logo_url**: "https://cdn.weatherapi.com/v4/images/weatherapi_logo.png"
- **contact_email**: "hello@contact.com"
- **legal_info_url**: "hello@legal.com"

## API Key

The tool requires an API key from WeatherAPI. You can sign up for a free account at https://www.weatherapi.com/, create a new API key, and add it to environment variables.

## Endpoint

The tool provides the following endpoints:

- **/get_weather_today**: Get today's weather. The input should be a location string.
- **/forecast_weather**: Forecast weather in the upcoming days. The input should be a location string and the number of days for the forecast.

## Function Descriptions

- **get_weather_today(location: str) -> str**: This function gets today's weather for a given location. The location should be a string. The function returns a string with the weather information.
- **forecast_weather(location: str, days: int) -> str**: This function forecasts the weather for a given location in the upcoming days. The location should be a string and days should be an integer. The function returns a string with the weather forecast.