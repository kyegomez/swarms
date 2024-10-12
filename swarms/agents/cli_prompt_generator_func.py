import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.prompts.prompt_generator_optimizer import (
    prompt_generator_sys_prompt,
)
from dotenv import load_dotenv
from swarms.agents.prompt_generator_agent import PromptGeneratorAgent
from yaml import dump

load_dotenv()


def generate_prompt(
    num_loops: int = 1,
    autosave: bool = True,
    save_to_yaml: bool = False,
    prompt: str = None,
    save_format: str = "yaml",
) -> None:
    """
    This function creates and runs a prompt generator agent with default settings for number of loops and autosave.

    Args:
        num_loops (int, optional): The number of loops to run the agent. Defaults to 1.
        autosave (bool, optional): Whether to autosave the agent's state. Defaults to True.
        save_to_yaml (bool, optional): Whether to save the agent's configuration to a YAML file. Defaults to False.
        prompt (str): The prompt to generate.
        save_format (str, optional): The format in which to save the generated prompt. Defaults to "yaml".

    Returns:
        None
    """
    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Create an instance of the OpenAIChat class
    model = OpenAIChat(
        openai_api_key=api_key,
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000,
    )

    # Initialize the agent
    agent = Agent(
        agent_name="Prompt-Optimizer",
        system_prompt=prompt_generator_sys_prompt.get_prompt(),
        llm=model,
        max_loops=num_loops,
        autosave=autosave,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        saved_state_path="optimizer_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
        context_length=200000,
        return_step_meta=False,
        output_type="string",
    )

    # Main Class
    prompt_generator = PromptGeneratorAgent(agent)

    # Run the agent
    prompt_generator.run(prompt, save_format)

    if save_to_yaml:
        with open("agent_config.yaml", "w") as file:
            dump(agent.config, file)


# # Example usage
# if __name__ == "__main__":
#     try:
#         create_and_run_agent(
#             num_loops=1,
#             autosave=True,
#             save_to_yaml=True,
#             prompt="Generate an amazing prompt for analyzing healthcare insurance documents",
#         )
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
