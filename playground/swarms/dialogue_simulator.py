from swarms import DialogueSimulator, Worker

worker1 = Worker(ai_name="Plinus", openai_api_key="")
worker2 = Worker(ai_name="Optimus Prime", openai_api_key="")

collab = DialogueSimulator(
    [worker1, worker2],
    # DialogueSimulator.select_next_speaker
)

collab.run(
    max_iters=4,
    name="plinus",
    message="how can we enable multi agent collaboration",
)
