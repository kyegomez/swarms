#boss node -> worker agent -> omni agent [worker of the worker]
from langchain.tools import tool
# from swarms.workers.multi_modal_workers.omni_agent.omni_chat import chat_huggingface
from swarms.workers.multi_modal_workers.omni_agent.omni_chat import chat_huggingface

class OmniWorkerAgent:
    def __init__(self, api_key, api_endpoint, api_type):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.api_type = api_type

    @tool
    def chat(self, data):
        """Chat with omni-modality model that uses huggingface to query for a specific model at run time. Translate text to speech, create images and more"""
        messages = data.get("messages")
        api_key = data.get("api_key", self.api_key)
        api_endpoint = data.get("api_endpoint", self.api_endpoint)
        api_type = data.get("api_type", self.api_type)

        if not(api_key and api_type and api_endpoint):
            raise ValueError("Please provide api_key, api_type, and api_endpoint")
        
        response = chat_huggingface(messages, api_key, api_type, api_endpoint)
        return response

