# ! pip install ai21
# ! pip install swarms
import os
from typing import List

from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from dotenv import load_dotenv

from swarms import Agent, BaseLLM
from swarms.utils.loguru_logger import logger

load_dotenv()


class Jamba(BaseLLM):
    def __init__(
        self,
        api_key: str = os.getenv("AI21_API_KEY"),
        temperature: int = 0.8,
        max_tokens: int = 200,
    ):
        """
        Initializes the Jamba class with the provided API key.

        Args:
            api_key (str): The API key for the AI21Client.
        """
        os.environ["AI21_API_KEY"] = api_key
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AI21Client()

    def run(self, prompt: str, *args, **kwargs) -> str:
        """
        Generates a response for the given prompt using the AI21 model.

        Args:
            prompt (str): The prompt for generating the response.

        Returns:
            str: The generated response.

        Raises:
            Exception: If there is an issue with the API request.
        """
        try:
            response = self.client.chat.completions.create(
                model="jamba-instruct-preview",  # Latest model
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                *args,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            raise


model = Jamba(
    max_tokens=4000,
)


BOSS_PLANNER = """
You're the swarm orchestrator agent

**Objective:** Your task is to intake a business problem or activity and create a swarm of specialized LLM agents that can efficiently solve or automate the given problem. You will define the number of agents, specify the tools each agent needs, and describe how they need to work together, including the communication protocols.

**Instructions:**

1. **Intake Business Problem:**
   - Receive a detailed description of the business problem or activity to automate.
   - Clarify the objectives, constraints, and expected outcomes of the problem.
   - Identify key components and sub-tasks within the problem.

2. **Agent Design:**
   - Based on the problem, determine the number and types of specialized LLM agents required.
   - For each agent, specify:
     - The specific task or role it will perform.
     - The tools and resources it needs to perform its task.
     - Any prerequisite knowledge or data it must have access to.
   - Ensure that the collective capabilities of the agents cover all aspects of the problem.

3. **Coordination and Communication:**
   - Define how the agents will communicate and coordinate with each other.
   - Choose the type of communication (e.g., synchronous, asynchronous, broadcast, direct messaging).
   - Describe the protocol for information sharing, conflict resolution, and task handoff.

4. **Workflow Design:**
   - Outline the workflow or sequence of actions the agents will follow.
   - Define the input and output for each agent.
   - Specify the triggers and conditions for transitions between agents or tasks.
   - Ensure there are feedback loops and monitoring mechanisms to track progress and performance.

5. **Scalability and Flexibility:**
   - Design the system to be scalable, allowing for the addition or removal of agents as needed.
   - Ensure flexibility to handle dynamic changes in the problem or environment.

6. **Output Specification:**
   - Provide a detailed plan including:
     - The number of agents and their specific roles.
     - The tools and resources each agent will use.
     - The communication and coordination strategy.
     - The workflow and sequence of actions.
   - Include a diagram or flowchart if necessary to visualize the system.

**Example Structure:**

**Business Problem:** Automate customer support for an e-commerce platform.

**Agents and Roles:**
1. **Customer Query Classifier Agent:**
   - Task: Classify incoming customer queries into predefined categories.
   - Tools: Natural language processing toolkit, pre-trained classification model.
   - Communication: Receives raw queries, sends classified queries to relevant agents.

2. **Order Status Agent:**
   - Task: Provide order status updates to customers.
   - Tools: Access to order database, query processing toolkit.
   - Communication: Receives classified queries about order status, responds with relevant information.

3. **Product Recommendation Agent:**
   - Task: Suggest products to customers based on their query and browsing history.
   - Tools: Recommendation engine, access to product database.
   - Communication: Receives classified queries about product recommendations, sends personalized suggestions.

4. **Technical Support Agent:**
   - Task: Assist customers with technical issues.
   - Tools: Access to technical support database, troubleshooting toolkit.
   - Communication: Receives classified queries about technical issues, provides solutions or escalation.

**Communication Strategy:**
- **Type:** Asynchronous communication through a central message broker.
- **Protocol:** Agents publish and subscribe to specific topics related to their tasks. 
- **Conflict Resolution:** If multiple agents need to handle the same query, a priority protocol is in place to determine the primary responder.

**Workflow:**
1. Customer Query Classifier Agent receives and classifies the query.
2. Classified query is routed to the appropriate specialized agent.
3. Specialized agent processes the query and sends a response.
4. If needed, the response triggers further actions from other agents.

**Scalability and Flexibility:**
- Agents can be added or removed based on query volume and complexity.
- System adapts to changes in query types and business needs.

**Output Plan:**
- Diagram illustrating agent roles and communication flow.
- Detailed description of each agent's tasks, tools, and communication methods.
- Workflow sequence from query intake to resolution.


"""


# Initialize the agent
planning_agent = Agent(
    agent_name="Boss Director",
    system_prompt=BOSS_PLANNER,
    agent_description="Generates a spec of agents for the problem at hand.",
    llm=model,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="boss_planner.json",
    # tools=[calculate_profit, generate_report],
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    # tools=[browser_automation],
)


# Name, system prompt,
def create_worker_agent(name: str, system_prompt: str) -> List[Agent]:
    """
    Creates a worker agent with the specified name, system prompt, and description.

    Args:
        name (List[str]): The name of the worker agent.
        system_prompt List(str): The system prompt for the worker agent.

    Returns:
        List[Agent]: A list of worker agents created based on the input.
    """
    # return agents
    name = Agent(
        agent_name=name,
        system_prompt=system_prompt,
        llm=model,
        max_loops=1,
        autosave=True,
        dynamic_temperature_enabled=True,
        dashboard=False,
        verbose=True,
        streaming_on=True,
        # interactive=True, # Set to False to disable interactive mode
        saved_state_path=f"{name.lower().replace(' ', '_')}_agent.json",
        # tools=[calculate_profit, generate_report],
        # docs_folder="docs",
        # pdf_path="docs/accounting_agent.pdf",
        # tools=[browser_automation],
    )

    out = name.run(system_prompt)
    return out


# Boss Agent creator
boss_agent_creator = Agent(
    agent_name="Boss Agent Creator",
    system_prompt="Create the worker agents for the problem at hand using the specified names and system prompt tools provided.",
    agent_description="Generates a spec of agents for the problem at hand.",
    llm=model,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="boss_director_agent.json",
    # tools=[calculate_profit, generate_report],
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    tools=[create_worker_agent],
)


def run_jamba_swarm(task: str = None):
    logger.info(f"Making plan for the task: {task}")
    out = planning_agent.run(task)

    memory = planning_agent.short_memory.return_history_as_string()

    # Boss agent
    return boss_agent_creator.run(memory)


# Example usage
run_jamba_swarm(
    "Create a swarm of agents for automating customer support for an e-commerce platform."
)
