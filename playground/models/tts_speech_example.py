from swarms import OpenAITTS
import os
from dotenv import load_dotenv

load_dotenv()

tts = OpenAITTS(
    model_name="tts-1-1106",
    voice="onyx",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_org_id=os.getenv("OPENAI_ORG_ID"),
)

out = tts.run_and_save("Dammmmmm those tacos were good")
print(out)
