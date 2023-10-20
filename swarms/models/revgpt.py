import argparse
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

    def enable_plugin(self, plugin_id: str):
        self.chatbot.install_plugin(plugin_id=plugin_id)

    def list_plugins(self):
        return self.chatbot.get_plugins()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage RevChatGPT plugins.')
    parser.add_argument('--enable', metavar='plugin_id', help='the plugin to enable')
    parser.add_argument('--list', action='store_true', help='list all available plugins')
    parser.add_argument('--access_token', required=True, help='access token for RevChatGPT')

    args = parser.parse_args()

    model = RevChatGPTModel(access_token=args.access_token)

    if args.enable:
        model.enable_plugin(args.enable)
    if args.list:
        plugins = model.list_plugins()
        for plugin in plugins:
            print(f"Plugin ID: {plugin['id']}, Name: {plugin['name']}")
