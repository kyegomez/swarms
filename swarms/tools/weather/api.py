import requests
import json
from ..tool import Tool
import os


def build_tool(config) -> Tool:
    tool = Tool(
        "Weather Info",
        "Look up weather information",
        name_for_model="Weather",
        description_for_model="Plugin for look up weather information",
        logo_url="https://cdn.weatherapi.com/v4/images/weatherapi_logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    KEY = config["subscription_key"]
    URL_CURRENT_WEATHER = "http://api.weatherapi.com/v1/current.json"
    URL_FORECAST_WEATHER = "http://api.weatherapi.com/v1/forecast.json"

    @tool.get("/get_weather_today")
    def get_weather_today(location: str):
        """Get today's the weather"""
        param = {"key": KEY, "q": location}
        res_completion = requests.get(URL_CURRENT_WEATHER, params=param)
        data = json.loads(res_completion.text.strip())
        output = {}
        output["overall"] = f"{data['current']['condition']['text']},\n"
        output["name"] = f"{data['location']['name']},\n"
        output["region"] = f"{data['location']['region']},\n"
        output["country"] = f"{data['location']['country']},\n"
        output["localtime"] = f"{data['location']['localtime']},\n"
        output[
            "temperature"
        ] = f"{data['current']['temp_c']}(C), {data['current']['temp_f']}(F),\n"
        output[
            "percipitation"
        ] = f"{data['current']['precip_mm']}(mm), {data['current']['precip_in']}(inch),\n"
        output["pressure"] = f"{data['current']['pressure_mb']}(milibar),\n"
        output["humidity"] = f"{data['current']['humidity']},\n"
        output["cloud"] = f"{data['current']['cloud']},\n"
        output[
            "body temperature"
        ] = f"{data['current']['feelslike_c']}(C), {data['current']['feelslike_f']}(F),\n"
        output[
            "wind speed"
        ] = f"{data['current']['gust_kph']}(kph), {data['current']['gust_mph']}(mph),\n"
        output[
            "visibility"
        ] = f"{data['current']['vis_km']}(km), {data['current']['vis_miles']}(miles),\n"
        output["UV index"] = f"{data['current']['uv']},\n"

        text_output = (
            f"Today's weather report for {data['location']['name']} is:\n"
            + "".join([f"{key}: {output[key]}" for key in output.keys()])
        )
        return text_output

    @tool.get("/forecast_weather")
    def forecast_weather(location: str, days: int):
        """Forecast weather in the upcoming days."""
        param = {
            "key": KEY,
            "q": location,
            "days": int(days),
        }
        res_completion = requests.get(URL_FORECAST_WEATHER, params=param)
        res_completion = json.loads(res_completion.text.strip())
        MAX_DAYS = 14
        res_completion = res_completion["forecast"]["forecastday"][
            int(days) - 1 if int(days) < MAX_DAYS else MAX_DAYS - 1
        ]
        output_dict = {}
        for k, v in res_completion["day"].items():
            output_dict[k] = v
        for k, v in res_completion["astro"].items():
            output_dict[k] = v

        output = {}
        output["over all weather"] = f"{output_dict['condition']['text']},\n"
        output[
            "max temperature"
        ] = f"{output_dict['maxtemp_c']}(C), {output_dict['maxtemp_f']}(F),\n"
        output[
            "min temperature"
        ] = f"{output_dict['mintemp_c']}(C), {output_dict['mintemp_f']}(F),\n"
        output[
            "average temperature"
        ] = f"{output_dict['avgtemp_c']}(C), {output_dict['avgtemp_f']}(F),\n"
        output[
            "max wind speed"
        ] = f"{output_dict['maxwind_kph']}(kph), {output_dict['maxwind_mph']}(mph),\n"
        output[
            "total precipitation"
        ] = f"{output_dict['totalprecip_mm']}(mm), {output_dict['totalprecip_in']}(inch),\n"
        output["will rain today"] = f"{output_dict['daily_will_it_rain']},\n"
        output["chance of rain"] = f"{output_dict['daily_chance_of_rain']},\n"
        output["total snow"] = f"{output_dict['totalsnow_cm']}(cm),\n"
        output["will snow today"] = f"{output_dict['daily_will_it_snow']},\n"
        output["chance of snow"] = f"{output_dict['daily_chance_of_snow']},\n"
        output[
            "average visibility"
        ] = f"{output_dict['avgvis_km']}(km), {output_dict['avgvis_miles']}(miles),\n"
        output["average humidity"] = f"{output_dict['avghumidity']},\n"
        output["UV index"] = f"{output_dict['uv']},\n"
        output["sunrise time"] = f"{output_dict['sunrise']},\n"
        output["sunset time"] = f"{output_dict['sunset']},\n"
        output["moonrise time"] = f"{output_dict['moonrise']},\n"
        output["moonset time"] = f"{output_dict['moonset']},\n"

        text_output = (
            f"The weather forecast for {param['q']} at {param['days']} days later is: \n"
            + "".join([f"{key}: {output[key]}" for key in output.keys()])
        )
        return text_output

    return tool
