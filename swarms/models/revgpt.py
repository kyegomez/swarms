import os
import revChatGPT
from revChatGPT.V1 import Chatbot as RevChatGPTV1, time
from revChatGPT.V3 import Chatbot as RevChatGPTV3

from abc import ABC, abstractmethod
from revChatGPT.V1 import Chatbot

class RevChatGPTModel:
    def __init__(self, access_token=None, **kwargs):
        super().__init__()
        self.config = kwargs
        if access_token:
            self.chatbot = Chatbot(config={"access_token": access_token})
        else:
            raise ValueError("access_token must be provided.")

    def run(self, task: str) -> str:
        self.start_time = time.time()
        prev_text = ""
        for data in self.chatbot.ask(task):
            message = data["message"][len(prev_text):]
            prev_text = data["message"]
        self.end_time = time.time()
        return prev_text

    def generate_summary(self, text: str) -> str:
        # Implement this method based on your requirements
        pass
