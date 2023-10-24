import os
from swarms.models.bing_chat import BingChat
from apps.discord import Bot
from dotenv import load_dotenv

load_dotenv()

# Initialize the EdgeGPTModel
cookie = os.environ.get("BING_COOKIE")
auth = os.environ.get("AUTH_COOKIE")
bing = BingChat(cookies_path="./cookies.json")

bot = Bot(llm=bing)
bot.generate_image(imggen=bing.create_img(auth_cookie=cookie, auth_cookie_SRCHHPGUSR=auth))
bot.send_text(use_agent=False)
