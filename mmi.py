from swarms import DialogueSimulator, Worker

worker = Worker(ai_name="Optimus Prime", openai_api_key="")

collab = DialogueSimulator(worker, DialogueSimulator.select_next_speaker)

collab.start("My name is Plinus and I am a worker", "How are you?")