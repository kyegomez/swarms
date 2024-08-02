import requests
import os

# API endpoint
url = "https://swarms.world/api/add-agent"  # replace with your actual API endpoint

# API key
api_key = os.getenv("SWARMS_API_KEY")  # replace with your actual API key

# Agent data
agent_data = {
    "name": "Sample Agent",
    "agent": "SampleAgent001",
    "language": "Python",
    "description": "This is a sample agent description.",
    "requirements": [
        {"package": "numpy", "installation": "pip install numpy"},
        {"package": "pandas", "installation": "pip install pandas"},
    ],
    "useCases": [
        {
            "title": "Data Analysis",
            "description": "Analyzes data using advanced algorithms.",
        },
        {
            "title": "Prediction",
            "description": "Predicts outcomes based on data.",
        },
    ],
    "tags": "data,analysis,prediction",
}

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# Sending POST request
response = requests.post(url, json=agent_data, headers=headers)

# Check response
if response.status_code == 200:
    print("Agent created successfully!")
else:
    print(f"Failed to create agent: {response.status_code}")
    print(response.json())
