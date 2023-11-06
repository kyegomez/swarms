from swarms.models.gpt4v import GPT4Vision

gpt4vision = GPT4Vision(api_key="")
task = "What is the following image about?"
img = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"

answer = gpt4vision.run(task, img)
