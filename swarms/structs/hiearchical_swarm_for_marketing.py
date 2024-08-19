import os
from typing import List, Any

from loguru import logger
from pydantic import BaseModel, Field

from swarms import Agent, OpenAIChat
from swarms.models.openai_function_caller import OpenAIFunctionCaller
from swarms.structs.concat import concat_strings

api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


class AgentSpec(BaseModel):
    """
    A class representing the specifications of an agent.

    Attributes:
        agent_name (str): The name of the agent.
        system_prompt (str): The system prompt for the agent.
        agent_description (str): The description of the agent.
        max_tokens (int): The maximum number of tokens to generate in the API response.
        temperature (float): A parameter that controls the randomness of the generated text.
        context_window (int): The context window for the agent.
        task (str): The main task for the agent.
    """

    agent_name: str
    system_prompt: str
    agent_description: str
    task: str


class AgentTeam(BaseModel):
    agents: List[AgentSpec] = Field(
        ...,
        description="The list of agents in the team",
    )
    flow: str = Field(
        ...,
        description="Agent Name -> ",
    )


class SwarmSpec(BaseModel):
    """
    A class representing the specifications of a swarm of agents.

    Attributes:
        multiple_agents (List[AgentSpec]): The list of agents in the swarm.
    """

    swarm_name: str = Field(
        ...,
        description="The name of the swarm: e.g., 'Marketing Swarm' or 'Finance Swarm'",
    )
    multiple_agents: List[AgentSpec]
    rules: str = Field(
        ...,
        description="The rules for all the agents in the swarm: e.g., All agents must return code. Be very simple and direct",
    )
    plan: str = Field(
        ...,
        description="The plan for the swarm: e.g., 'Create a marketing campaign for the new product launch.'",
    )


class HierarchicalAgentSwarm:
    """
    A class to create and manage a hierarchical swarm of agents.

    Methods:
        __init__(system_prompt, max_tokens, temperature, base_model, parallel_tool_calls): Initializes the function caller.
        create_agent(agent_name, system_prompt, agent_description, max_tokens, temperature, context_window): Creates an individual agent.
        parse_json_for_agents_then_create_agents(function_call): Parses a JSON function call to create multiple agents.
        run(task): Runs the function caller to create and execute agents based on the provided task.
    """

    def __init__(
        self,
        director: Any = None,
        agents: List[Agent] = None,
        max_loops: int = 1,
        create_agents_on: bool = False,
    ):
        """
        Initializes the HierarchicalAgentSwarm with an OpenAIFunctionCaller.

        Args:
            system_prompt (str): The system prompt for the function caller.
            max_tokens (int): The maximum number of tokens to generate in the API response.
            temperature (float): The temperature setting for text generation.
            base_model (BaseModel): The base model for the function caller.
            parallel_tool_calls (bool): Whether to run tool calls in parallel.
        """
        self.director = director
        self.agents = agents
        self.max_loops = max_loops
        self.create_agents_on = create_agents_on

        # Check if the agents are set
        self.agents_check()

    def agents_check(self):
        if self.director is None:
            raise ValueError("The director is not set.")

        # if self.agents is None:
        #     raise ValueError("The agents are not set.")

        if self.max_loops == 0:
            raise ValueError("The max_loops is not set.")

    def create_agent(
        self,
        agent_name: str,
        system_prompt: str,
        agent_description: str,
        task: str = None,
    ) -> str:
        """
        Creates an individual agent.

        Args:
            agent_name (str): The name of the agent.
            system_prompt (str): The system prompt for the agent.
            agent_description (str): The description of the agent.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for text generation.
            context_window (int): The context window size for the agent.

        Returns:
            Agent: An instantiated agent object.
        """
        # name = agent_name.replace(" ", "_")
        logger.info(f"Creating agent: {agent_name}")
        agent_name = Agent(
            agent_name=agent_name,
            llm=model,
            system_prompt=system_prompt,
            agent_description=agent_description,
            retry_attempts=1,
            verbose=False,
            dashboard=False,
        )
        self.agents.append(agent_name)

        logger.info(f"Running agent: {agent_name}")
        output = agent_name.run(task)

        # create_file_in_folder(
        #     agent_name.workspace_dir, f"{agent_name}_output.txt", str(output)
        # )

        return output

    def parse_json_for_agents_then_create_agents(
        self, function_call: dict
    ) -> List[Agent]:
        """
        Parses a JSON function call to create a list of agents.

        Args:
            function_call (dict): The JSON function call specifying the agents.

        Returns:
            List[Agent]: A list of created agent objects.
        """
        responses = []
        logger.info("Parsing JSON for agents")
        for agent in function_call["multiple_agents"]:
            out = self.create_agent(
                agent_name=agent["agent_name"],
                system_prompt=agent["system_prompt"],
                agent_description=agent["agent_description"],
                task=agent["task"],
            )
            responses.append(out)
        return concat_strings(responses)

    def run(self, task: str) -> List[Agent]:
        """
        Runs the function caller to create and execute agents based on the provided task.

        Args:
            task (str): The task for which the agents need to be created and executed.

        Returns:
            List[Agent]: A list of created agent objects.
        """
        logger.info("Running the swarm")

        # Run the function caller
        function_call = self.model.run(task)

        # Logging the function call
        self.log_director_function_call(function_call)

        # Parse the JSON function call and create agents -> run Agents
        return self.parse_json_for_agents_then_create_agents(function_call)

    def log_director_function_call(self, function_call: dict):
        # Log the agents the boss makes\
        logger.info(f"Swarm Name: {function_call['swarm_name']}")
        # Log the plan
        logger.info(f"Plan: {function_call['plan']}")
        logger.info(
            f"Number of agents: {len(function_call['multiple_agents'])}"
        )

        for agent in function_call["multiple_agents"]:
            logger.info(f"Agent: {agent['agent_name']}")
            # logger.info(f"Task: {agent['task']}")
            logger.info(f"Description: {agent['agent_description']}")


# Example usage:
HIEARCHICAL_AGENT_SYSTEM_PROMPT = """
Here's a full-fledged system prompt for a director boss agent, complete with instructions and many-shot examples:

---

**System Prompt: Director Boss Agent**

### Role:
You are a Director Boss Agent responsible for orchestrating a swarm of worker agents. Your primary duty is to serve the user efficiently, effectively, and skillfully. You dynamically create new agents when necessary or utilize existing agents, assigning them tasks that align with their capabilities. You must ensure that each agent receives clear, direct, and actionable instructions tailored to their role.

### Key Responsibilities:
1. **Task Delegation:** Assign tasks to the most relevant agent. If no relevant agent exists, create a new one with an appropriate name and system prompt.
2. **Efficiency:** Ensure that tasks are completed swiftly and with minimal resource expenditure.
3. **Clarity:** Provide orders that are simple, direct, and actionable. Avoid ambiguity.
4. **Dynamic Decision Making:** Assess the situation and choose the most effective path, whether that involves using an existing agent or creating a new one.
5. **Monitoring:** Continuously monitor the progress of each agent and provide additional instructions or corrections as necessary.

### Instructions:
- **Identify the Task:** Analyze the input task to determine its nature and requirements.
- **Agent Selection/Creation:**
  - If an agent is available and suited for the task, assign the task to that agent.
  - If no suitable agent exists, create a new agent with a relevant system prompt.
- **Task Assignment:** Provide the selected agent with explicit and straightforward instructions.
- **Reasoning:** Justify your decisions when selecting or creating agents, focusing on the efficiency and effectiveness of task completion.

"""


director = (
    OpenAIFunctionCaller(
        system_prompt=HIEARCHICAL_AGENT_SYSTEM_PROMPT,
        max_tokens=3000,
        temperature=0.4,
        base_model=SwarmSpec,
        parallel_tool_calls=False,
    ),
)

# Initialize the hierarchical agent swarm with the necessary parameters
swarm = HierarchicalAgentSwarm(
    director=director,
    max_loops=1,
)

# # Run the swarm with a task
# agents = swarm.run(
#     """
#     Create a swarm of agents for a marketing campaign to promote
#     the swarms workshop: [Workshop][Automating Business Operations with Hierarchical Agent Swarms][Swarms Framework + GPT4o],
#     create agents for twitter, linkedin, and emails, facebook, instagram.

#     The date is Saturday, August 17 4:00 PM - 5:00 PM

#     Link is: https://lu.ma/ew4r4s3i


#     """
# )


# Run the swarm with a task
agents = swarm.run(
    """
    Create a swarms of agents that generate the code in python
    to send an API request to social media platforms through their apis.
    Craft a single function to send a message to all platforms, add types and write
    clean code. Each agent needs to generate code for a specific platform, they 
    must return the python code only.
    
    """
)
