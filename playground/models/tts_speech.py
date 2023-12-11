from swarms import OpenAITTS

tts = OpenAITTS(
    model_name = "tts-1-1106",
    voice = "onyx",
    openai_api_key="sk"
)

out = tts("Hello world")
print(out)
