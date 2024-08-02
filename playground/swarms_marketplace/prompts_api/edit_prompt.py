import requests
import json
import os

url = "https://swarms.world/api/edit-prompt"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('SWARMS_API_KEY')}"
}

data = {
    "id": "prompt_id",
    "name": "Updated Prompt",
    "prompt": "This is an updated prompt from an API route.",
    "description": "Updated description of the prompt.",
    "useCases": [
        {"title": "Updated use case 1", "description": "Updated description of use case 1"},
        {"title": "Updated use case 2", "description": "Updated description of use case 2"}
    ],
    "tags": "updated, prompt"
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())