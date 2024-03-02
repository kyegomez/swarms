from swarms.agents import MultiModalAgent

load_dict = {"ImageCaptioning": "cuda"}

node = MultiModalAgent(load_dict)

text = node.run_text(
    "What is your name? Generate a picture of yourself"
)

img = node.run_img("/image1", "What is this image about?")

chat = node.chat(
    "What is your name? Generate a picture of yourself. What is"
    " this image about?",
    streaming=True,
)
