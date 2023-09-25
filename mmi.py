from swarms import DialogueSimulator, Worker

collab = DialogueSimulator(Worker, DialogueSimulator.select_next_speaker)
collab.start("My name is Plinus and I am a worker", "How are you?")

