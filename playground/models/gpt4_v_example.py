from swarms.models.gpt4v import GPT4Vision

gpt4vision = GPT4Vision(openai_api_key="")

img = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/VFPt_Solenoid_correct2.svg/640px-VFPt_Solenoid_correct2.svg.png"

task = "What is this image"

answer = gpt4vision.run(task, img)

print(answer)
