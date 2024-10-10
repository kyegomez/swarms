import os
from swarms.structs.agent import Agent
from swarm_models.popular_llms import OpenAIChat
from swarms.structs.agent_registry import AgentRegistry

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


# Registry of agents
agent_registry = AgentRegistry(
    name="Swarms CLI",
    description="A registry of agents for the Swarms CLI",
)


def create_agent(name: str, system_prompt: str, max_loops: int = 1):
    """
    Create and initialize an agent with the given parameters.

    Args:
        name (str): The name of the agent.
        system_prompt (str): The system prompt for the agent.
        max_loops (int, optional): The maximum number of loops the agent can perform. Defaults to 1.

    Returns:
        Agent: The initialized agent.

    """
    # Initialize the agent
    agent = Agent(
        agent_name=name,
        system_prompt=system_prompt,
        llm=model,
        max_loops=max_loops,
        autosave=True,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        saved_state_path=f"{name}.json",
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        # return_step_meta=True,
        # disable_print_every_step=True,
        # output_type="json",
        interactive=True,
    )

    agent_registry.add(agent)

    return agent


# Run the agents in the registry
def run_agent_by_name(name: str, task: str, *args, **kwargs):
    """
    Run an agent by its name and perform a specified task.

    Parameters:
    - name (str): The name of the agent.
    - task (str): The task to be performed by the agent.
    - *args: Variable length argument list.
    - **kwargs: Arbitrary keyword arguments.

    Returns:
    - output: The output of the agent's task.

    """
    agent = agent_registry.get_agent_by_name(name)

    output = agent.run(task, *args, **kwargs)

    return output


# # Test
# out = create_agent("Accountant1", "Prepares financial statements")
# print(out)
