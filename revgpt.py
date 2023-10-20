import os
import sys
from dotenv import load_dotenv
from swarms.models.revgptV4 import RevChatGPTModelv4
from swarms.models.revgptV1 import RevChatGPTModelv1

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

load_dotenv()

config = {
    "model": os.getenv("REVGPT_MODEL"),
    "plugin_ids": [os.getenv("REVGPT_PLUGIN_IDS")],
    "disable_history": os.getenv("REVGPT_DISABLE_HISTORY") == "True",
    "PUID": os.getenv("REVGPT_PUID"),
    "unverified_plugin_domains": [os.getenv("REVGPT_UNVERIFIED_PLUGIN_DOMAINS")]
}

# For v1 model
model = RevChatGPTModelv1(access_token=os.getenv("ACCESS_TOKEN"), **config)
# model = RevChatGPTModelv4(access_token=os.getenv("ACCESS_TOKEN"), **config)

# For v3 model
# model = RevChatGPTModel(access_token=os.getenv("OPENAI_API_KEY"), **config)

task = "Write a cli snake game"
response = model.run(task)
print(response)
