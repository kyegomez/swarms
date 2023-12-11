from swarms import OpenAITTS

tts = OpenAITTS(
    model_name="tts-1-1106",
    voice="onyx",
    openai_api_key="YOUR_API_KEY",
)

out = tts.run_and_save("pliny is a girl and a chicken")
print(out)
