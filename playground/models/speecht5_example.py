from swarms.models.speecht5 import SpeechT5Wrapper

speechT5 = SpeechT5Wrapper()

result = speechT5("Hello, how are you?")

speechT5.save_speech(result)
print("Speech saved successfully!")
