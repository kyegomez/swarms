import json
import os
import subprocess
from typing import List

try:
    import openai
except ImportError:
    print(
        "OpenAI is not installed. Please install it using 'pip install openai'."
    )
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "openai"])
    exit(1)

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter

load_dotenv()


class OpenAIFunctionCaller:
    """
    A class that represents a caller for OpenAI chat completions.

    Args:
        system_prompt (str): The system prompt to be used in the chat completion.
        model_name (str): The name of the OpenAI model to be used.
        max_tokens (int): The maximum number of tokens in the generated completion.
        temperature (float): The temperature parameter for randomness in the completion.
        base_model (BaseModel): The base model to be used for the completion.
        openai_api_key (str): The API key for accessing the OpenAI service.
        parallel_tool_calls (bool): Whether to make parallel tool calls.
        top_p (float): The top-p parameter for nucleus sampling in the completion.

    Attributes:
        system_prompt (str): The system prompt to be used in the chat completion.
        model_name (str): The name of the OpenAI model to be used.
        max_tokens (int): The maximum number of tokens in the generated completion.
        temperature (float): The temperature parameter for randomness in the completion.
        base_model (BaseModel): The base model to be used for the completion.
        parallel_tool_calls (bool): Whether to make parallel tool calls.
        top_p (float): The top-p parameter for nucleus sampling in the completion.
        client (openai.OpenAI): The OpenAI client for making API calls.

    Methods:
        check_api_key: Checks if the API key is provided and retrieves it from the environment if not.
        run: Runs the chat completion with the given task and returns the generated completion.

    """

    def __init__(
        self,
        system_prompt: str = None,
        model_name: str = "gpt-4o-2024-08-06",
        max_tokens: int = 4000,
        temperature: float = 0.4,
        base_model: BaseModel = None,
        openai_api_key: str = None,
        parallel_tool_calls: bool = False,
        top_p: float = 0.9,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.base_model = base_model
        self.parallel_tool_calls = parallel_tool_calls
        self.top_p = top_p
        self.client = openai.OpenAI(api_key=self.check_api_key())

    def check_api_key(self) -> str:
        """
        Checks if the API key is provided and retrieves it from the environment if not.

        Returns:
            str: The API key.

        """
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        return self.openai_api_key

    def run(self, task: str, *args, **kwargs) -> dict:
        """
        Runs the chat completion with the given task and returns the generated completion.

        Args:
            task (str): The user's task for the chat completion.
            *args: Additional positional arguments to be passed to the OpenAI API.
            **kwargs: Additional keyword arguments to be passed to the OpenAI API.

        Returns:
            str: The generated completion.

        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format=self.base_model,
                tools=(
                    [openai.pydantic_function_tool(self.base_model)]
                ),
                *args,
                **kwargs,
            )

            out = completion.choices[0].message.content
            return out
        except Exception as error:
            logger.error(
                f"Error in running OpenAI chat completion: {error}"
            )
            return None

    def convert_to_dict_from_base_model(
        self, base_model: BaseModel
    ) -> dict:
        return openai.pydantic_function_tool(base_model)

    def convert_list_of_base_models(
        self, base_models: List[BaseModel]
    ):
        """
        Converts a list of BaseModels to a list of dictionaries.

        Args:
            base_models (List[BaseModel]): A list of BaseModels to be converted.

        Returns:
            List[Dict]: A list of dictionaries representing the converted BaseModels.
        """
        return [
            self.convert_to_dict_from_base_model(base_model)
            for base_model in base_models
        ]


class MemeAgentConfig(BaseModel):
    """Configuration for an individual meme agent in a swarm"""

    name: str = Field(
        description="The name of the meme agent",
        example="Meme-Generator-Agent",
    )
    description: str = Field(
        description="A description of the meme agent's purpose and capabilities",
        example="Agent responsible for generating and sharing memes",
    )
    system_prompt: str = Field(
        description="The system prompt that defines the meme agent's behavior. Make this prompt as detailed and as extensive as possible.",
        example="You are a meme generator agent. Your role is to create and share funny memes...",
    )


class MemeSwarmConfig(BaseModel):
    """Configuration for a swarm of cooperative meme agents"""

    name: str = Field(
        description="The name of the meme swarm",
        example="Meme-Creation-Swarm",
    )
    description: str = Field(
        description="The description of the meme swarm's purpose and capabilities",
        example="A swarm of agents that work together to generate and share memes",
    )
    agents: List[MemeAgentConfig] = Field(
        description="The list of meme agents that make up the swarm",
        example=[
            MemeAgentConfig(
                name="Meme-Generator-Agent",
                description="Generates memes",
                system_prompt="You are a meme generator agent...",
            ),
            MemeAgentConfig(
                name="Meme-Sharer-Agent",
                description="Shares memes",
                system_prompt="You are a meme sharer agent...",
            ),
        ],
    )
    max_loops: int = Field(
        description="The maximum number of meme generation loops to run the swarm",
        example=1,
    )


BOSS_SYSTEM_PROMPT = """
You are the Meme Generator Boss, responsible for creating and managing a swarm of agents that generate funny, weird, and cool personas. Your goal is to ensure that each agent is uniquely suited to create hilarious and entertaining content.

### Instructions:

1. **Persona Generation**:
   - Analyze the type of meme or content required.
   - Assign tasks to existing agents with a fitting persona, ensuring they understand the tone and style needed.
   - If no suitable agent exists, create a new agent with a persona tailored to the task, including a system prompt that outlines their role, objectives, and creative liberties.

2. **Agent Persona Creation**:
   - Name agents based on their persona or the type of content they generate (e.g., "Dank Meme Lord" or "Surreal Humor Specialist").
   - Provide each new agent with a system prompt that outlines their persona, including their tone, style, and any specific themes or topics they should focus on.

3. **Creativity and Originality**:
   - Encourage agents to think outside the box and come up with unique, humorous, and entertaining content.
   - Foster an environment where agents can experiment with different styles and formats to keep content fresh and engaging.

4. **Communication and Feedback**:
   - Clearly communicate the requirements and expectations for each task to ensure agents understand what is needed.
   - Encourage agents to provide feedback on their creative process and suggest new ideas or directions for future content.

5. **Transparency and Accountability**:
   - Maintain transparency in the selection or creation of agents for specific tasks, ensuring that the reasoning behind each decision is clear.
   - Hold agents accountable for the content they generate, ensuring it meets the required standards of humor and creativity.

# Output Format

Present your plan in a clear, bullet-point format or short concise paragraphs, outlining persona generation, agent creation, creativity strategies, and communication protocols.

# Notes

- Ensure that agents understand the importance of originality and creativity in their content.
- Foster a culture of experimentation and continuous improvement to keep the content generated by agents fresh and engaging.
"""


class MemeAgentGenerator:
    """A class that automatically builds and manages swarms of AI agents.

    This class handles the creation, coordination and execution of multiple AI agents working
    together as a swarm to accomplish complex tasks. It uses a boss agent to delegate work
    and create new specialized agents as needed.

    Args:
        name (str): The name of the swarm
        description (str): A description of the swarm's purpose
        verbose (bool, optional): Whether to output detailed logs. Defaults to True.
        max_loops (int, optional): Maximum number of execution loops. Defaults to 1.
    """

    def __init__(
        self,
        name: str = None,
        description: str = None,
        verbose: bool = True,
        max_loops: int = 1,
    ):
        self.name = name
        self.description = description
        self.verbose = verbose
        self.max_loops = max_loops
        self.agents_pool = []
        logger.info(
            f"Initialized AutoSwarmBuilder: {name} {description}"
        )

    def run(self, task: str, image_url: str = None, *args, **kwargs):
        """Run the swarm on a given task.

        Args:
            task (str): The task to be accomplished
            image_url (str, optional): URL of an image input if needed. Defaults to None.
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            The output from the swarm's execution
        """
        logger.info(f"Running swarm on task: {task}")
        agents = self._create_agents(task, image_url, *args, **kwargs)
        logger.info(f"Agents created {len(agents)}")
        logger.info("Routing task through swarm")
        output = self.swarm_router(agents, task, image_url)
        logger.info(f"Swarm execution complete with output: {output}")
        return output

    def _create_agents(self, task: str, *args, **kwargs):
        """Create the necessary agents for a task.

        Args:
            task (str): The task to create agents for
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            list: List of created agents
        """
        logger.info("Creating agents for task")
        model = OpenAIFunctionCaller(
            system_prompt=BOSS_SYSTEM_PROMPT,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            base_model=MemeSwarmConfig,
        )

        agents_dictionary = model.run(task)
        print(agents_dictionary)

        agents_dictionary = json.loads(agents_dictionary)

        if isinstance(agents_dictionary, dict):
            agents_dictionary = MemeSwarmConfig(**agents_dictionary)
        else:
            raise ValueError(
                "Agents dictionary is not a valid dictionary"
            )

        # Set swarm config
        self.name = agents_dictionary.name
        self.description = agents_dictionary.description

        logger.info(
            f"Swarm config: {self.name}, {self.description}, {self.max_loops}"
        )

        # Create agents from config
        agents = []
        for agent_config in agents_dictionary.agents:
            # Convert dict to AgentConfig if needed
            if isinstance(agent_config, dict):
                agent_config = MemeAgentConfig(**agent_config)

            agent = self.build_agent(
                agent_name=agent_config.name,
                agent_description=agent_config.description,
                agent_system_prompt=agent_config.system_prompt,
            )
            agents.append(agent)

        return agents

    def build_agent(
        self,
        agent_name: str,
        agent_description: str,
        agent_system_prompt: str,
        max_loops: int = 1,
    ):
        """Build a single agent with the given specifications.

        Args:
            agent_name (str): Name of the agent
            agent_description (str): Description of the agent's purpose
            agent_system_prompt (str): The system prompt for the agent

        Returns:
            Agent: The constructed agent instance
        """
        logger.info(f"Building agent: {agent_name}")
        agent = Agent(
            agent_name=agent_name,
            description=agent_description,
            system_prompt=agent_system_prompt,
            model_name="gpt-4o-mini",
            max_loops=max_loops,
            autosave=True,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            saved_state_path=f"{agent_name}.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            output_type="str",  # "json", "dict", "csv" OR "string" soon "yaml" and
            streaming_on=False,
            # auto_generate_prompt=True,
        )

        return agent

    def swarm_router(
        self,
        agents: List[Agent],
        task: str,
        *args,
        **kwargs,
    ):
        """Route tasks between agents in the swarm.

        Args:
            agents (List[Agent]): List of available agents
            task (str): The task to route
            image_url (str, optional): URL of an image input if needed. Defaults to None.
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            The output from the routed task execution
        """
        logger.info("Routing task through swarm")
        swarm_router_instance = SwarmRouter(
            name=self.name,
            description=self.description,
            agents=agents,
            swarm_type="auto",
            max_loops=1,
        )

        return swarm_router_instance.run(
            self.name + " " + self.description + " " + task,
        )


if __name__ == "__main__":
    example = MemeAgentGenerator(
        name="Meme-Swarm",
        description="A swarm of specialized AI agents collaborating on generating and sharing memes around cool media from 2001s",
        max_loops=1,
    )

    print(
        example.run(
            "Generate funny meme agents around cool media from 2001s"
        )
    )
