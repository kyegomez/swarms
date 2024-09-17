import requests


# Fetch all prompts with optional filters
def get_prompts(filters):
    response = requests.get(
        "https://swarms.world/get-prompts", params=filters
    )

    if response.status_code != 200:
        raise Exception(
            f"Error: {response.status_code}, {response.text}"
        )

    data = response.json()
    print(data)


# Fetch prompt by ID
def get_prompt_by_id(id):
    response = requests.get(f"https://swarms.world/get-prompts/{id}")

    if response.status_code != 200:
        raise Exception(
            f"Error: {response.status_code}, {response.text}"
        )

    data = response.json()
    print(data)


# Example usage
get_prompts(
    {
        "name": "example",
        "tag": "tag1,tag2",
        "use_case": "example",
        "use_case_description": "description",
    }
)
get_prompt_by_id("123")
