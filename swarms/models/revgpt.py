import os
import revChatGPT
from revChatGPT.V1 import Chatbot as RevChatGPTV1
from revChatGPT.V3 import Chatbot as RevChatGPTV3

class RevChatGPTModel:
    def __init__(self, access_token=None, api_key=None, **kwargs):
        self.config = kwargs
        if access_token:
            self.chatbot = RevChatGPTV1(config={"access_token": access_token})
        elif api_key:
            self.chatbot = RevChatGPTV3(api_key=api_key)
        else:
            raise ValueError("Either access_token or api_key must be provided.")

    def run(self, task: str) -> str:
        response = ""
        for data in self.chatbot.ask(task):
            response = data["message"]
        return response

    def generate_summary(self, text: str) -> str:
        # Implement summary generation using RevChatGPT
        pass
