from swarms.agents.reasoning_duo import ReasoningDuo

if __name__ == "__main__":
    # Initialize the ReasoningDuo with two lightweight models
    duo = ReasoningDuo(
        model_names=["gpt-4o-mini", "gpt-4o-mini"],
        # max_loops=1,  # Remove this line
    )

    # Batched tasks to process
    tasks = [
        "Summarize the benefits of solar energy.",
        "List three uses of robotics in healthcare.",
    ]

    # Run the batch once and print each result
    results = duo.batched_run(tasks)
    for task, output in zip(tasks, results):
        print(f"Task: {task}\nResult: {output}\n")
