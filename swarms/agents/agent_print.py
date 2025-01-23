from swarms.utils.formatter import formatter


def agent_print(
    agent_name: str,
    response: str = None,
    loop_count: int = None,
    streaming_on: bool = False,
):
    """
    Prints the response from an agent based on the streaming mode.

    Args:
        agent_name (str): The name of the agent.
        response (str): The response from the agent.
        loop_count (int): The maximum number of loops.
        streaming_on (bool): Indicates if streaming is on or off.

    Returns:
        str: The response from the agent.
    """
    if streaming_on:
        formatter.print_panel_token_by_token(
            f"{agent_name}: {response}",
            title=f"Agent Name: {agent_name} [Max Loops: {loop_count}]",
        )
    else:
        formatter.print_panel(
            f"{agent_name}: {response}",
            f"Agent Name {agent_name} [Max Loops: {loop_count} ]",
        )

    return response
