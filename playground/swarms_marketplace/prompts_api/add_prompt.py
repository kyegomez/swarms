import requests
import json
import os

url = "https://swarms.world/api/add-prompt"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('SWARMS_API_KEY')}"
}

data = {
    "name": "Example Prompt",
    "prompt": "This is an example prompt from an API route.",
    "description": "Description of the prompt.",
    "useCases": [
        {"title": "Use case 1", "description": "Description of use case 1"},
        {"title": "Use case 2", "description": "Description of use case 2"}
    ],
    "tags": "example, prompt"
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())