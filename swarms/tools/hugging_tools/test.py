from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "hugging_tools", "http://127.0.0.1:8079/tools/hugging_tools/"
tools_name, tools_config = load_single_tools(tool_name, tool_url)
print(tools_name, tools_config)

qa = STQuestionAnswerer()
print(tools_config)
agent = qa.load_tools(tools_name, tools_config)

# # NLP task
# # Translation
# answer = agent("Please translate this into English: Bonjour!")
# print(answer)

# # Sentence similarity
# answer = agent("Which of these two sentences has a similar meaning to the other? Sentence 1: The cat sat on the mat. Sentence 2: The feline rested on the floor.")
# print(answer)

# # Question answering
# answer = agent("What is the capital of France?")
# print(answer)

# # Text classification
# answer = agent("Which category does this text belong to? This book is a thrilling mystery novel.")
# print(answer)

# # Token classification
# answer = agent("Which words in this sentence are key words? The quick brown fox jumped over the lazy dog.")
# print(answer)

# # Text2Text generation
# answer = agent("Please generate a paragraph about the benefits of exercise using the following prompt: 'Regular exercise can help improve'")
# print(answer)

# # Summarization
# answer = agent("""Please provide a summary of the article with the following passage:

# China, officially known as the People's Republic of China, is one of the largest countries in the world. It is located in East Asia, bordered by 14 countries and the East China Sea, Yellow Sea, South China Sea, and Taiwan Strait. With a population of over 1.4 billion people, China is the most populous country in the world.

# """)
# print(answer)

# # Conversational
# answer = agent("Can you generate a response to the following prompt in a conversational manner? 'What do you think about the weather today?'")
# print(answer)

# # Text generation
# answer = agent("Please generate a paragraph of text about artificial intelligence using the following prompt: 'Artificial intelligence is'")
# print(answer)

# # Audio task
# # automatic speech recognition
# answer = agent("What does the man say in the audio `test.flac`?")
# print(answer)

# # audio to audio
# answer = agent("Enhance this speech `test.flac` more clearly.")
# print(answer)

# # text to speech & audio to audio
# answer = agent("Generate a audio where a man says: hello, world!, and then enchance the generated audio.")
# print(answer)

# # audio classification
# answer = agent("classify this audio: `test.flac`")
# print(answer)

# # CV task
# # text-to-image
# answer = agent("Please generate a picture with prompt: a cat.")
# print(answer)

# # image-to-text
# answer = agent("What does the picture `boat.png` shows?")
# print(answer)

# # image-segmentation
# answer = agent("Please divide the `boat.png` into proper segments using appropriate models.")
# print(answer)

# # object-detection
# answer = agent("Detect the objects in the picture `boat.png`.")
# print(answer)

# # image-classification
# answer = agent("Classify the picture `boat.png`.")
# print(answer)

# # visual-question-answering
# answer = agent("Answer the question according to the photo `boat.png`: what is it?")
# print(answer)

# # document-question-answering
# answer = agent("Answer the question based on the content of the document screenshot `doc.jpg`: What is the topic of this document?(hint: use the document-question-answering)")
# print(answer)
