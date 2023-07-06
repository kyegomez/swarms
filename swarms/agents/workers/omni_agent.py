from langchain import tool
from swarms.agents.workers.multi_modal.omni_agent import chat_huggingface

class OmniWorkerAgent:
    def __init__(self, api_key, api_endpoint, api_type):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.api_type = api_type

    @tool
    def chat(self, data):
        messages = data.get("messages")
        api_key = data.get("api_key", self.api_key)
        api_endpoint = data.get("api_endpoint", self.api_endpoint)
        api_type = data.get("api_type", self.api_type)

        if not(api_key and api_type and api_endpoint):
            raise ValueError("Please provide api_key, api_type, and api_endpoint")
        
        response = self.chat_huggingface(messages, api_key, api_type, api_endpoint)
        return response
    


#usage
agent = OmniWorkerAgent(api_key="your key", api_endpoint="api endpoint", api_type="you types")

data = {
    "messages": "your_messages",
    "api_key": "your_api_key",
    "api_endpoint": "your_api_endpoint",
    "api_type": "your_api_type"
}

response = agent.chat(data)


print(response)  # Prints the response
