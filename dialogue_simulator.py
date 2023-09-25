from swarms import DialogueSimulator, Worker

worker1 = Worker(ai_name="Plinus", openai_api_key="")
worker2 = Worker(ai_name="Optimus Prime", openai_api_key="")

collab = DialogueSimulator([worker1, worker2], DialogueSimulator.select_next_speaker)
collab.reset()

# collab.start("My name is Plinus and I am a worker", "How are you?")

max_iters = 6
n = 0

while n < max_iters:
    name, message = collab.step()
    print(f"({name}): {message}")
    print("\n")
    n += 1