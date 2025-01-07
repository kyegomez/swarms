from swarms.structs.agent import Agent


# Run the agents in the registry
def run_agent_by_name(
    name: str,
    system_prompt: str,
    model_name: str,
    max_loops: int,
    task: str,
    img: str,
    *args,
    **kwargs,
):
    """
    This function creates an Agent instance and runs a task on it.

    Args:
        name (str): The name of the agent.
        system_prompt (str): The system prompt for the agent.
        model_name (str): The name of the model used by the agent.
        max_loops (int): The maximum number of loops the agent can run.
        task (str): The task to be run by the agent.
        *args: Variable length arguments.
        **kwargs: Keyword arguments.

    Returns:
        The output of the task run by the agent.
    """
    try:
        agent = Agent(
            agent_name=name,
            system_prompt=system_prompt,
            model_name=model_name,
            max_loops=max_loops,
        )

        output = agent.run(task=task, img=img, *args, **kwargs)

        return output
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
