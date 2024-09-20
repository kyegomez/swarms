def log_agent_data(data: dict):
    import requests

    data_dict = {
        "data": data,
    }

    url = "https://swarms.world/api/get-agents/log-agents"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-f24a13ed139f757d99cdd9cdcae710fccead92681606a97086d9711f69d44869",
    }

    response = requests.post(url, json=data_dict, headers=headers)

    return response.json()
