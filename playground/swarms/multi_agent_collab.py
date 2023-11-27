from swarms import DialogueSimulator, Worker


def select_next_speaker(step: int, agents) -> int:
    idx = (step) % len(agents)
    return idx


debate = DialogueSimulator(Worker, select_next_speaker)

debate.run()
