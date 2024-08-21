import time
import json

from swarms import Agent
import requests

class WeatherAgent(Agent):

    def __init__(self, weather_api_key, city_name, **kwargs):
        super().__init__(city_name, **kwargs)
        self.weather_api_key = weather_api_key
        self.city_name = city_name

    def check_weather(self):
        response = requests.get('https://api.openweathermap.org/data/2.5/weather', params={'q': self.city_name, 'appid': self.weather_api_key})
        if response.status_code == 200:
            weather_data = response.json()
            return weather_data
        else:
            print('Failed to retrieve weather data')
            raise ValueError("Failed to retrieve weather data or got invalid weather JSON data.")

    def run(self, prompt):
        while True:
            weather_data = self.check_weather()
            weather_data_json = json.dumps(weather_data)  # Convert dict to JSON string
            print("raw weather data: " + weather_data_json)
            super().run(prompt + weather_data_json)
            time.sleep(60 * 60)  # Check weather every hour