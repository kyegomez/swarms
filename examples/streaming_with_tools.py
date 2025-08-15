from swarms import Agent

def get_weather(location: str, units: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location (str): The city/location to get weather for
        units (str): Temperature units (celsius or fahrenheit)

    Returns:
        str: Weather information
    """
    # Simulated weather data
    weather_data = {
        "New York": {"temperature": "22°C", "condition": "sunny", "humidity": "65%"},
        "London": {"temperature": "15°C", "condition": "cloudy", "humidity": "80%"},
        "Tokyo": {"temperature": "28°C", "condition": "rainy", "humidity": "90%"},
    }
    
    location_key = location.title()
    if location_key in weather_data:
        data = weather_data[location_key]
        temp = data["temperature"] 
        if units == "fahrenheit" and "°C" in temp:
            # Convert to Fahrenheit for demo
            celsius = int(temp.replace("°C", ""))
            fahrenheit = (celsius * 9/5) + 32
            temp = f"{fahrenheit}°F"
        
        return f"Weather in {location}: {temp}, {data['condition']}, humidity: {data['humidity']}"
    else:
        return f"Weather data not available for {location}"
    
agent = Agent(
    model_name="gpt-4o",
    max_loops=1,
    verbose=True,
    streaming_on=True,
    print_on=True,
    tools=[get_weather],
)

agent.run("What is the weather in Tokyo? ")