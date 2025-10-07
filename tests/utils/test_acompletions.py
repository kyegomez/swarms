from litellm import completion
from dotenv import load_dotenv

load_dotenv()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Retrieve detailed current weather information for a specified location, including temperature, humidity, wind speed, and atmospheric conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA, or a specific geographic coordinate in the format 'latitude,longitude'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit", "kelvin"],
                        "description": "The unit of temperature measurement to be used in the response.",
                    },
                    "include_forecast": {
                        "type": "boolean",
                        "description": "Indicates whether to include a short-term weather forecast along with the current conditions.",
                    },
                    "time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional parameter to specify the time for which the weather data is requested, in ISO 8601 format.",
                    },
                },
                "required": [
                    "location",
                    "unit",
                    "include_forecast",
                    "time",
                ],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston today?",
    }
]


response = completion(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=True,
)

print(response.choices[0].message.tool_calls[0].function.arguments)
print(response.choices[0].message)
