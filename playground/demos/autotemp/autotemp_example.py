from autotemp import AutoTemp

# Your OpenAI API key
api_key = ""

autotemp_agent = AutoTemp(
    api_key=api_key,
    alt_temps=[0.4, 0.6, 0.8, 1.0, 1.2],
    auto_select=False,
    # model_version="gpt-3.5-turbo"  # Specify the model version if needed
)

# Define the task and temperature string
task = "Generate a short story about a lost civilization."
temperature_string = "0.4,0.6,0.8,1.0,1.2,"

# Run the AutoTempAgent
result = autotemp_agent.run(task, temperature_string)

# Print the result
print(result)
