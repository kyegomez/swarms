import os
from dotenv import load_dotenv
from swarms.models.revgptV4 import RevChatGPTModel
from swarms.workers.worker import Worker

load_dotenv()

config = {
    "model": os.getenv("REVGPT_MODEL"),
    "plugin_ids": [os.getenv("REVGPT_PLUGIN_IDS")],
    "disable_history": os.getenv("REVGPT_DISABLE_HISTORY") == "True",
    "PUID": os.getenv("REVGPT_PUID"),
    "unverified_plugin_domains": [os.getenv("REVGPT_UNVERIFIED_PLUGIN_DOMAINS")],
}

llm = RevChatGPTModel(access_token=os.getenv("ACCESS_TOKEN"), **config)

worker = Worker(ai_name="Optimus Prime", llm=llm)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = worker.run(task)
print(response)
