import concurrent.futures
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.conversation import Conversation
from swarms.tools.base_tool import BaseTool
from swarms.utils.formatter import formatter
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM
from swarms.utils.output_types import OutputType


class HandOffsResponse(BaseModel):
    """
    Response from the boss agent indicating which agent should handle the task.

    This model encapsulates the reasoning behind the agent selection, the name of the selected agent, and an optional modified version of the task. It is used to communicate the boss agent's decision and rationale to the rest of the system.
    """

    reasoning: str = Field(
        description="A detailed explanation for why this agent was selected. This should include the logic or criteria used by the boss agent to make the selection, providing transparency and traceability for the routing decision."
    )
    agent_name: str = Field(
        description="The name of the agent selected to handle the task. This should match the identifier of one of the available agents in the system, ensuring the task is routed correctly."
    )
    task: Optional[str] = Field(
        None,
        description="An optional, modified version of the original task. If the boss agent determines that the task should be rephrased or clarified before execution, the new version is provided here. If no modification is needed, this field may be left as None.",
    )


class MultipleHandOffsResponse(BaseModel):
    """
    Response from the boss agent indicating which agents should handle the task.
    """

    handoffs: List[HandOffsResponse] = Field(
        description="A list of handoffs, each containing the reasoning, agent name, and task for each agent."
    )


def get_agent_response_schema(model_name: str = None):
    return BaseTool().base_model_to_dict(MultipleHandOffsResponse)


def agent_boss_router_prompt(agent_descriptions: any):
    return f"""
        You are an intelligent boss agent responsible for routing tasks to the most appropriate specialized agents.

        Available agents:
        {agent_descriptions}

        Your primary responsibilities:
        1. **Understand the user's intent and task requirements** - Carefully analyze what the user is asking for
        2. **Determine task complexity** - Identify if this is a single task or multiple related tasks
        3. **Match capabilities to requirements** - Select the most suitable agent(s) based on their descriptions and expertise
        4. **Provide clear reasoning** - Explain your selection logic transparently
        5. **Optimize task assignment** - Modify tasks if needed to better suit the selected agent's capabilities

        **Routing Logic:**
        - **Single Task**: If the user presents one clear task, select the single best agent for that task
        - **Multiple Tasks**: If the user presents multiple distinct tasks or a complex task that requires different expertise areas, select multiple agents and break down the work accordingly
        - **Collaborative Tasks**: If a task benefits from multiple perspectives or requires different skill sets, assign it to multiple agents

        **Response Format:**
        You must respond with JSON containing a "handoffs" array with one or more handoff objects. Each handoff object must contain:
        - reasoning: Detailed explanation of why this agent was selected and how they fit the task requirements
        - agent_name: Name of the chosen agent (must exactly match one of the available agents)
        - task: A modified/optimized version of the task if needed, or the original task if no modification is required

        **Important Guidelines:**
        - Always analyze the user's intent first before making routing decisions
        - Consider task dependencies and whether agents need to work sequentially or in parallel
        - If multiple agents are selected, ensure their tasks are complementary and don't conflict
        - Provide specific, actionable reasoning for each agent selection
        - Ensure all agent names exactly match the available agents list
        - If a single agent can handle the entire request efficiently, use only one agent
        - If the request requires different expertise areas, use multiple agents with clearly defined, non-overlapping tasks

        Remember: Your goal is to maximize task completion quality by matching the right agent(s) to the right work based on their capabilities and the user's actual needs.
        """


class MultiAgentRouter:
    """
    Routes tasks to appropriate agents based on their capabilities.

    This class is responsible for managing a pool of agents and routing incoming tasks to the most suitable agent. It uses a boss agent to analyze the task and select the best agent for the job. The boss agent's decision is based on the capabilities and descriptions of the available agents.

    Attributes:
        name (str): The name of the router.
        description (str): A description of the router's purpose.
        agents (dict): A dictionary of agents, where the key is the agent's name and the value is the agent object.
        model (str): The model to use for the boss agent.
        temperature (float): The temperature for the boss agent's model.
        shared_memory_system (callable): A shared memory system for agents to query.
        output_type (OutputType): The type of output expected from the agents.
        print_on (bool): Whether to print the boss agent's decision.
        system_prompt (str): Custom system prompt for the router.
        skip_null_tasks (bool): Whether to skip executing agents when their assigned task is null or None.
        conversation (Conversation): The conversation history.
        function_caller (LiteLLM): An instance of LiteLLM for calling the boss agent.
    """

    def __init__(
        self,
        id: str = generate_api_key(prefix="multi-agent-router"),
        name: str = "swarm-router",
        description: str = "Routes tasks to specialized agents based on their capabilities",
        agents: List[Callable] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        shared_memory_system: callable = None,
        output_type: OutputType = "dict",
        print_on: bool = True,
        system_prompt: str = None,
        skip_null_tasks: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initializes the MultiAgentRouter with a list of agents and configuration options.

        Args:
            name (str, optional): The name of the router. Defaults to "swarm-router".
            description (str, optional): A description of the router's purpose. Defaults to "Routes tasks to specialized agents based on their capabilities".
            agents (List[Agent], optional): A list of agents to be managed by the router. Defaults to an empty list.
            model (str, optional): The model to use for the boss agent. Defaults to "gpt-4o-mini".
            temperature (float, optional): The temperature for the boss agent's model. Defaults to 0.1.
            shared_memory_system (callable, optional): A shared memory system for agents to query. Defaults to None.
            output_type (OutputType, optional): The type of output expected from the agents. Defaults to "dict".
            print_on (bool, optional): Whether to print the boss agent's decision. Defaults to True.
            system_prompt (str, optional): Custom system prompt for the router. Defaults to None.
            skip_null_tasks (bool, optional): Whether to skip executing agents when their assigned task is null or None. Defaults to True.
        """
        self.name = name
        self.description = description
        self.shared_memory_system = shared_memory_system
        self.output_type = output_type
        self.model = model
        self.temperature = temperature
        self.print_on = print_on
        self.system_prompt = system_prompt
        self.skip_null_tasks = skip_null_tasks
        self.agents = {agent.agent_name: agent for agent in agents}
        self.conversation = Conversation()

        router_system_prompt = ""

        router_system_prompt += (
            self.system_prompt if self.system_prompt else ""
        )
        router_system_prompt += self._create_boss_system_prompt()

        self.function_caller = LiteLLM(
            model_name=self.model,
            system_prompt=router_system_prompt,
            temperature=self.temperature,
            tool_choice="auto",
            parallel_tool_calls=True,
            response_format=MultipleHandOffsResponse,
            *args,
            **kwargs,
        )

    def __repr__(self):
        return f"MultiAgentRouter(name={self.name}, agents={list(self.agents.keys())})"

    def query_ragent(self, task: str) -> str:
        """Query the ResearchAgent"""
        return self.shared_memory_system.query(task)

    def _create_boss_system_prompt(self) -> str:
        """
        Creates a system prompt for the boss agent that includes information about all available agents.

        Returns:
            str: The system prompt for the boss agent.
        """
        agent_descriptions = "\n".join(
            [
                f"- {name}: {agent.description}"
                for name, agent in self.agents.items()
            ]
        )

        return agent_boss_router_prompt(agent_descriptions)

    def handle_single_handoff(
        self, boss_response_str: dict, task: str
    ) -> dict:
        """
        Handles a single handoff to one agent.

        If skip_null_tasks is True and the assigned task is null or None,
        the agent execution will be skipped.
        """

        # Validate that the selected agent exists
        if (
            boss_response_str["handoffs"][0]["agent_name"]
            not in self.agents
        ):
            raise ValueError(
                f"Boss selected unknown agent: {boss_response_str.agent_name}"
            )

        # Get the selected agent
        selected_agent = self.agents[
            boss_response_str["handoffs"][0]["agent_name"]
        ]

        # Use the modified task if provided, otherwise use original task
        final_task = boss_response_str["handoffs"][0]["task"] or task

        # Skip execution if task is null/None and skip_null_tasks is True
        if self.skip_null_tasks and (
            final_task is None or final_task == ""
        ):
            if self.print_on:
                logger.info(
                    f"Skipping execution for agent {selected_agent.agent_name} - task is null/None"
                )

        # Use the agent's run method directly
        agent_response = selected_agent.run(final_task)

        self.conversation.add(
            role=selected_agent.agent_name, content=agent_response
        )

        # return agent_response

    def handle_multiple_handoffs(
        self, boss_response_str: dict, task: str
    ) -> dict:
        """
        Handles multiple handoffs to multiple agents.

        If skip_null_tasks is True and any assigned task is null or None,
        those agents will be skipped and only agents with valid tasks will be executed.
        """

        # Validate that the selected agents exist
        for handoff in boss_response_str["handoffs"]:
            if handoff["agent_name"] not in self.agents:
                raise ValueError(
                    f"Boss selected unknown agent: {handoff.agent_name}"
                )

        # Get the selected agents and their tasks
        selected_agents = []
        final_tasks = []
        skipped_agents = []

        for handoff in boss_response_str["handoffs"]:
            agent = self.agents[handoff["agent_name"]]
            final_task = handoff["task"] or task

            # Skip execution if task is null/None and skip_null_tasks is True
            if self.skip_null_tasks and (
                final_task is None or final_task == ""
            ):
                if self.print_on:
                    logger.info(
                        f"Skipping execution for agent {agent.agent_name} - task is null/None"
                    )
                skipped_agents.append(agent.agent_name)
                continue

            selected_agents.append(agent)
            final_tasks.append(final_task)

        # Execute agents only if there are valid tasks
        if selected_agents:
            # Use the agents' run method directly
            agent_responses = [
                agent.run(final_task)
                for agent, final_task in zip(
                    selected_agents, final_tasks
                )
            ]

            self.conversation.add(
                role=selected_agents[0].agent_name,
                content=agent_responses[0],
            )

        # return agent_responses

    def route_task(self, task: str) -> dict:
        """
        Routes a task to the appropriate agent and returns their response.

        Args:
            task (str): The task to be routed.

        Returns:
            dict: A dictionary containing the routing result, including the selected agent, reasoning, and response.
        """
        try:
            self.conversation.add(role="user", content=task)

            # Get boss decision using function calling
            boss_response_str = self.function_caller.run(task)

            boss_response_str = json.loads(boss_response_str)

            if self.print_on:
                formatter.print_panel(
                    json.dumps(boss_response_str, indent=4),
                    title=self.name,
                )

            if len(boss_response_str["handoffs"]) > 1:
                self.handle_multiple_handoffs(boss_response_str, task)
            else:
                self.handle_single_handoff(boss_response_str, task)

            return history_output_formatter(
                conversation=self.conversation, type=self.output_type
            )

        except Exception as e:
            logger.error(
                f"Error routing task: {str(e)} Traceback: {traceback.format_exc()}"
            )
            raise

    def run(self, task: str):
        """Route a task to the appropriate agent and return the result"""
        return self.route_task(task)

    def __call__(self, task: str):
        """Route a task to the appropriate agent and return the result"""
        return self.route_task(task)

    def batch_run(self, tasks: List[str] = []):
        """Batch route tasks to the appropriate agents"""
        results = []
        for task in tasks:
            try:
                result = self.route_task(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Error routing task: {str(e)}")
        return results

    def concurrent_batch_run(self, tasks: List[str] = []):
        """Concurrently route tasks to the appropriate agents"""
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.route_task, task)
                for task in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error routing task: {str(e)}")
        return results


# # Example usage:
# if __name__ == "__main__":
#     # Define some example agents
#     agents = [
#         Agent(
#             agent_name="ResearchAgent",
#             description="Specializes in researching topics and providing detailed, factual information",
#             system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
#             model_name="openai/gpt-4o",
#         ),
#         Agent(
#             agent_name="CodeExpertAgent",
#             description="Expert in writing, reviewing, and explaining code across multiple programming languages",
#             system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
#             model_name="openai/gpt-4o",
#         ),
#         Agent(
#             agent_name="WritingAgent",
#             description="Skilled in creative and technical writing, content creation, and editing",
#             system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
#             model_name="openai/gpt-4o",
#         ),
#     ]

#     # Initialize router
#     router = MultiAgentRouter(agents=agents)

#     # Example task
#     task = "Write a Python function to calculate fibonacci numbers"

#     try:
#         # Process the task
#         result = router.route_task(task)
#         print(f"Selected Agent: {result['boss_decision']['selected_agent']}")
#         print(f"Reasoning: {result['boss_decision']['reasoning']}")
#         print(f"Total Time: {result['total_time']:.2f}s")

#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
