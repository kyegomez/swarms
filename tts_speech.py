from swarms import OpenAITTS

tts = OpenAITTS(
    model_name = "tts-1-1106",
    voice = "onyx",
    openai_api_key="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"
)

out = tts("Hello world")
print(out)
