from swarms import Agent, ConcurrentWorkflow


def streaming_callback(
    agent_name: str, chunk: str, is_complete: bool
):
    """
    Print streaming output from each agent as it arrives.
    """
    if chunk:
        print(f"[{agent_name}] {chunk}", end="", flush=True)
    if is_complete:
        print(" [DONE]")


def main():
    """
    Run a simple concurrent workflow with streaming output.
    """
    agents = [
        Agent(
            agent_name="Financial",
            system_prompt="Financial analysis.",
            model_name="groq/moonshotai/kimi-k2-instruct",
            max_loops=1,
            print_on=False,
        ),
        Agent(
            agent_name="Legal",
            system_prompt="Legal analysis.",
            model_name="groq/moonshotai/kimi-k2-instruct",
            max_loops=1,
            print_on=False,
        ),
    ]

    workflow = ConcurrentWorkflow(
        name="Simple-Streaming",
        agents=agents,
        show_dashboard=False,
        output_type="dict-all-except-first",
    )

    task = "Give a short analysis of the risks and opportunities for a SaaS startup expanding to Europe."

    print("Streaming output:\n")
    result = workflow.run(
        task=task, streaming_callback=streaming_callback
    )
    print("\n\nFinal results:")
    # Convert list of messages to dict with agent names as keys
    agent_outputs = {}
    for message in result:
        if message["role"] != "User":  # Skip the user message
            agent_outputs[message["role"]] = message["content"]

    for agent, output in agent_outputs.items():
        print(f"{agent}: {output}")


if __name__ == "__main__":
    main()
