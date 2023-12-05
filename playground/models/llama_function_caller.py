from swarms.models.llama_function_caller import LlamaFunctionCaller

llama_caller = LlamaFunctionCaller()


# Add a custom function
def get_weather(location: str, format: str) -> str:
    # This is a placeholder for the actual implementation
    return f"Weather at {location} in {format} format."


llama_caller.add_func(
    name="get_weather",
    function=get_weather,
    description="Get the weather at a location",
    arguments=[
        {
            "name": "location",
            "type": "string",
            "description": "Location for the weather",
        },
        {
            "name": "format",
            "type": "string",
            "description": "Format of the weather data",
        },
    ],
)

# Call the function
result = llama_caller.call_function(
    "get_weather", location="Paris", format="Celsius"
)
print(result)

# Stream a user prompt
llama_caller("Tell me about the tallest mountain in the world.")
