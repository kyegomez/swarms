import os
from swarms.models.bing_chat import BingChat
from dotenv import load_dotenv

load_dotenv()

# Initialize the EdgeGPTModel
edgegpt = BingChat(cookies_path="./cookies.json")
cookie = os.environ.get("BING_COOKIE")
auth = os.environ.get("AUTH_COOKIE")

# Use the worker to process a task
task = "hi"
# img_task = "Sunset over mountains"

response = edgegpt(task)
# response = edgegpt.create_img(auth_cookie=cookie,auth_cookie_SRCHHPGUSR=auth,prompt=img_task)

print(response)
