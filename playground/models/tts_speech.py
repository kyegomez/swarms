from swarms import OpenAITTS
import os
from dotenv import load_dotenv

load_dotenv()

tts = OpenAITTS(
    model_name="tts-1-1106",
    voice="onyx",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

out = tts.run_and_save("Dammmmmm those tacos were good")
print(out)
