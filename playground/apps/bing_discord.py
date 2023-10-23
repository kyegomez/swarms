import os
from swarms.models.bing_chat import BingChat
from apps.discord import Bot
from dotenv import load_dotenv


# Initialize the EdgeGPTModel
cookie = os.environ.get("BING_COOKIE")
auth = os.environ.get("AUTH_COOKIE")
bing = BingChat(cookies_path="./cookies.txt", bing_cookie=cookie, auth_cookie=auth)

bot = Bot(llm=bing, cookie=cookie, auth=auth)
bot.generate_image(imggen=bing.create_img())
bot.send_text(use_agent=False)
