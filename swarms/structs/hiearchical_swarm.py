"""
Hierarchical Swarm Implementation

This module provides a hierarchical swarm architecture where a director agent coordinates
multiple worker agents to execute complex tasks through a structured workflow.

Flow:
1. User provides a task
2. Director creates a plan
3. Director distributes orders to agents individually or multiple tasks at once
4. Agents execute tasks and report back to the director
5. Director evaluates results and issues new orders if needed (up to max_loops)
6. All context and conversation history is preserved throughout the process

Todo

- Add layers of management -- a list of list of agents that act as departments
- Auto build agents from input prompt - and then add them to the swarm
- Create an interactive and dynamic UI like we did with heavy swarm
- Make it faster and more high performance

Classes:
    HierarchicalOrder: Represents a single task assignment to a specific agent
    SwarmSpec: Contains the overall plan and list of orders for the swarm
    HierarchicalSwarm: Main swarm orchestrator that manages director and worker agents
"""

import traceback
from typing import Any, Callable, List, Optional, Union

from pydantic import BaseModel, Field

from swarms.prompts.hiearchical_system_prompt import (
    HIEARCHICAL_SWARM_SYSTEM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import list_all_agents
from swarms.tools.base_tool import BaseTool
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="hierarchical_swarm")


class HierarchicalOrder(BaseModel):
    """
    Represents a single task assignment within the hierarchical swarm.

    This class defines the structure for individual task orders that the director
    distributes to worker agents. Each order specifies which agent should execute
    what specific task.

    Attributes:
        agent_name (str): The name of the agent assigned to execute the task.
                         Must match an existing agent in the swarm.
        task (str): The specific task description to be executed by the assigned agent.
                   Should be clear and actionable.
    """

    agent_name: str = Field(
        ...,
        description="Specifies the name of the agent to which the task is assigned. This is a crucial element in the hierarchical structure of the swarm, as it determines the specific agent responsible for the task execution.",
    )
    task: str = Field(
        ...,
        description="Defines the specific task to be executed by the assigned agent. This task is a key component of the swarm's plan and is essential for achieving the swarm's goals.",
    )


class SwarmSpec(BaseModel):
    """
    Defines the complete specification for a hierarchical swarm execution.

    This class contains the overall plan and all individual orders that the director
    creates to coordinate the swarm's activities. It serves as the structured output
    format for the director agent.

    Attributes:
        plan (str): A comprehensive plan outlining the sequence of actions and strategy
                   for the entire swarm to accomplish the given task.
        orders (List[HierarchicalOrder]): A list of specific task assignments to
                                         individual agents within the swarm.
    """

    plan: str = Field(
        ...,
        description="Outlines the sequence of actions to be taken by the swarm. This plan is a detailed roadmap that guides the swarm's behavior and decision-making.",
    )
    orders: List[HierarchicalOrder] = Field(
        ...,
        description="A collection of task assignments to specific agents within the swarm. These orders are the specific instructions that guide the agents in their task execution and are a key element in the swarm's plan.",
    )


class HierarchicalSwarm:
    """
    A hierarchical swarm orchestrator that coordinates multiple agents through a director.

    This class implements a hierarchical architecture where a director agent creates
    plans and distributes tasks to worker agents. The director can provide feedback
    and iterate on results through multiple loops to achieve the desired outcome.

    The swarm maintains conversation history throughout the process, allowing for
    context-aware decision making and iterative refinement of results.

    Attributes:
        name (str): The name identifier for this swarm instance.
        description (str): A description of the swarm's purpose and capabilities.
        director (Optional[Union[Agent, Callable, Any]]): The director agent that
                                                         coordinates the swarm.
        agents (List[Union[Agent, Callable, Any]]): List of worker agents available
                                                   for task execution.
        max_loops (int): Maximum number of feedback loops the swarm can perform.
        output_type (OutputType): Format for the final output of the swarm.
        feedback_director_model_name (str): Model name for the feedback director.
        director_name (str): Name identifier for the director agent.
        director_model_name (str): Model name for the main director agent.
        verbose (bool): Whether to enable detailed logging and progress tracking.
        add_collaboration_prompt (bool): Whether to add collaboration prompts to agents.
        planning_director_agent (Optional[Union[Agent, Callable, Any]]): Optional
                                                                        planning agent.
        director_feedback_on (bool): Whether director feedback is enabled.
    """

    def __init__(
        self,
        name: str = "HierarchicalAgentSwarm",
        description: str = "Distributed task swarm",
        director: Optional[Union[Agent, Callable, Any]] = None,
        agents: List[Union[Agent, Callable, Any]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict-all-except-first",
        feedback_director_model_name: str = "gpt-4o-mini",
        director_name: str = "Director",
        director_model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        add_collaboration_prompt: bool = True,
        planning_director_agent: Optional[
            Union[Agent, Callable, Any]
        ] = None,
        director_feedback_on: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize a new HierarchicalSwarm instance.

        Args:
            name (str): The name identifier for this swarm instance.
            description (str): A description of the swarm's purpose.
            director (Optional[Union[Agent, Callable, Any]]): The director agent.
                                                             If None, a default director will be created.
            agents (List[Union[Agent, Callable, Any]]): List of worker agents.
                                                       Must not be empty.
            max_loops (int): Maximum number of feedback loops (must be > 0).
            output_type (OutputType): Format for the final output.
            feedback_director_model_name (str): Model name for feedback director.
            director_name (str): Name identifier for the director agent.
            director_model_name (str): Model name for the main director agent.
            verbose (bool): Whether to enable detailed logging.
            add_collaboration_prompt (bool): Whether to add collaboration prompts.
            planning_director_agent (Optional[Union[Agent, Callable, Any]]):
                Optional planning agent for enhanced planning capabilities.
            director_feedback_on (bool): Whether director feedback is enabled.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If no agents are provided or max_loops is invalid.
        """
        self.name = name
        self.description = description
        self.director = director
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.feedback_director_model_name = (
            feedback_director_model_name
        )
        self.director_name = director_name
        self.verbose = verbose
        self.director_model_name = director_model_name
        self.add_collaboration_prompt = add_collaboration_prompt
        self.planning_director_agent = planning_director_agent
        self.director_feedback_on = director_feedback_on

        self.init_swarm()

    def init_swarm(self):
        """
        Initialize the swarm with proper configuration and validation.

        This method performs the following initialization steps:
        1. Sets up logging if verbose mode is enabled
        2. Creates a conversation instance for history tracking
        3. Performs reliability checks on the configuration
        4. Adds agent context to the director

        Raises:
            ValueError: If the swarm configuration is invalid.
        """
        # Initialize logger only if verbose is enabled
        if self.verbose:
            logger.info(
                f"🚀 Initializing HierarchicalSwarm: {self.name}"
            )

        self.conversation = Conversation(time_enabled=False)

        # Reliability checks
        self.reliability_checks()

        self.add_context_to_director()

        if self.verbose:
            logger.success(
                f"✅ HierarchicalSwarm: {self.name} initialized successfully."
            )

    def add_context_to_director(self):
        """
        Add agent context and collaboration information to the director's conversation.

        This method ensures that the director has complete information about all
        available agents, their capabilities, and how they can collaborate. This
        context is essential for the director to make informed decisions about
        task distribution.

        Raises:
            Exception: If adding context fails due to agent configuration issues.
        """
        try:
            if self.verbose:
                logger.info("📝 Adding agent context to director")

            list_all_agents(
                agents=self.agents,
                conversation=self.conversation,
                add_to_conversation=True,
                add_collaboration_prompt=self.add_collaboration_prompt,
            )

            if self.verbose:
                logger.success(
                    "✅ Agent context added to director successfully"
                )

        except Exception as e:
            error_msg = (
                f"❌ Failed to add context to director: {str(e)}"
            )
            logger.error(
                f"{error_msg}\n🔍 Traceback: {traceback.format_exc()}"
            )

    def setup_director(self):
        """
        Set up the director agent with proper configuration and tools.

        Creates a new director agent with the SwarmSpec schema for structured
        output, enabling it to create plans and distribute orders effectively.

        Returns:
            Agent: A configured director agent ready to coordinate the swarm.

        Raises:
            Exception: If director setup fails due to configuration issues.
        """
        try:
            if self.verbose:
                logger.info("🎯 Setting up director agent")

            schema = BaseTool().base_model_to_dict(SwarmSpec)

            if self.verbose:
                logger.debug(f"📋 Director schema: {schema}")

            return Agent(
                agent_name=self.director_name,
                agent_description="A director agent that can create a plan and distribute orders to agents",
                model_name=self.director_model_name,
                max_loops=1,
                base_model=SwarmSpec,
                tools_list_dictionary=[schema],
                output_type="dict-all-except-first",
            )

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def reliability_checks(self):
        """
        Perform validation checks to ensure the swarm is properly configured.

        This method validates:
        1. That at least one agent is provided
        2. That max_loops is greater than 0
        3. That a director is available (creates default if needed)

        Raises:
            ValueError: If the swarm configuration is invalid.
        """
        try:
            if self.verbose:
                logger.info(
                    f"Hiearchical Swarm: {self.name} Reliability checks in progress..."
                )

            if not self.agents or len(self.agents) == 0:
                raise ValueError(
                    "No agents found in the swarm. At least one agent must be provided to create a hierarchical swarm."
                )

            if self.max_loops <= 0:
                raise ValueError(
                    "Max loops must be greater than 0. Please set a valid number of loops."
                )

            if self.director is None:
                self.director = self.setup_director()

            if self.verbose:
                logger.success(
                    f"Hiearchical Swarm: {self.name} Reliability checks passed..."
                )

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def run_director(
        self,
        task: str,
        img: str = None,
    ) -> SwarmSpec:
        """
        Execute the director agent with the given task and conversation context.

        This method runs the director agent to create a plan and distribute orders
        based on the current task and conversation history. If a planning director
        agent is configured, it will first create a detailed plan before the main
        director processes the task.

        Args:
            task (str): The task to be executed by the director.
            img (str, optional): Optional image input for the task.

        Returns:
            SwarmSpec: The director's output containing the plan and orders.

        Raises:
            Exception: If director execution fails.
        """
        try:
            if self.verbose:
                logger.info(
                    f"🎯 Running director with task: {task[:100]}..."
                )

            if self.planning_director_agent is not None:
                plan = self.planning_director_agent.run(
                    task=f"History: {self.conversation.get_str()} \n\n Create a detailed step by step comprehensive plan for the director to execute the task: {task}",
                    img=img,
                )

                task += plan

            # Run the director with the context
            function_call = self.director.run(
                task=f"History: {self.conversation.get_str()} \n\n Task: {task}",
                img=img,
            )

            self.conversation.add(
                role="Director", content=function_call
            )

            if self.verbose:
                logger.success("✅ Director execution completed")
                logger.debug(
                    f"📋 Director output type: {type(function_call)}"
                )

            return function_call

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def step(self, task: str, img: str = None, *args, **kwargs):
        """
        Execute a single step of the hierarchical swarm workflow.

        This method performs one complete iteration of the swarm's workflow:
        1. Run the director to create a plan and orders
        2. Parse the director's output to extract plan and orders
        3. Execute all orders by calling the appropriate agents
        4. Optionally generate director feedback on the results

        Args:
            task (str): The task to be processed in this step.
            img (str, optional): Optional image input for the task.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The results from this step, either agent outputs or director feedback.

        Raises:
            Exception: If step execution fails.
        """
        try:
            if self.verbose:
                logger.info(
                    f"👣 Executing single step for task: {task[:100]}..."
                )

            output = self.run_director(task=task, img=img)

            # Parse the orders
            plan, orders = self.parse_orders(output)

            if self.verbose:
                logger.info(
                    f"📋 Parsed plan and {len(orders)} orders"
                )

            # Execute the orders
            outputs = self.execute_orders(orders)

            if self.verbose:
                logger.info(f"⚡ Executed {len(outputs)} orders")

            if self.director_feedback_on is True:
                feedback = self.feedback_director(outputs)
            else:
                feedback = outputs

            if self.verbose:
                logger.success("✅ Step completed successfully")

            return feedback

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def run(self, task: str, img: str = None, *args, **kwargs):
        """
        Execute the hierarchical swarm for the specified number of feedback loops.

        This method orchestrates the complete swarm execution, performing multiple
        iterations based on the max_loops configuration. Each iteration builds upon
        the previous results, allowing for iterative refinement and improvement.

        The method maintains conversation history throughout all loops and provides
        context from previous iterations to subsequent ones.

        Args:
            task (str): The initial task to be processed by the swarm.
            img (str, optional): Optional image input for the agents.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The formatted conversation history as output, formatted according
                 to the output_type configuration.

        Raises:
            Exception: If swarm execution fails.
        """
        try:
            current_loop = 0
            last_output = None

            if self.verbose:
                logger.info(
                    f"🚀 Starting hierarchical swarm run: {self.name}"
                )
                logger.info(
                    f"📊 Configuration - Max loops: {self.max_loops}"
                )

            while current_loop < self.max_loops:
                if self.verbose:
                    logger.info(
                        f"🔄 Loop {current_loop + 1}/{self.max_loops} - Processing task"
                    )

                # For the first loop, use the original task.
                # For subsequent loops, use the feedback from the previous loop as context.
                if current_loop == 0:
                    loop_task = task
                else:
                    loop_task = (
                        f"Previous loop results: {last_output}\n\n"
                        f"Original task: {task}\n\n"
                        "Based on the previous results and any feedback, continue with the next iteration of the task. "
                        "Refine, improve, or complete any remaining aspects of the analysis."
                    )

                # Execute one step of the swarm
                try:
                    last_output = self.step(
                        task=loop_task, img=img, *args, **kwargs
                    )

                    if self.verbose:
                        logger.success(
                            f"✅ Loop {current_loop + 1} completed successfully"
                        )

                except Exception as e:
                    error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
                    logger.error(error_msg)

                current_loop += 1

                # Add loop completion marker to conversation
                self.conversation.add(
                    role="System",
                    content=f"--- Loop {current_loop}/{self.max_loops} completed ---",
                )

            if self.verbose:
                logger.success(
                    f"🎉 Hierarchical swarm run completed: {self.name}"
                )
                logger.info(
                    f"📊 Total loops executed: {current_loop}"
                )

            return history_output_formatter(
                conversation=self.conversation, type=self.output_type
            )

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def feedback_director(self, outputs: list):
        """
        Generate feedback from the director based on agent outputs.

        This method creates a feedback director agent that analyzes the results
        from worker agents and provides specific, actionable feedback for improvement.
        The feedback is added to the conversation history and can be used in
        subsequent iterations.

        Args:
            outputs (list): List of outputs from worker agents that need feedback.

        Returns:
            str: The director's feedback on the agent outputs.

        Raises:
            Exception: If feedback generation fails.
        """
        try:
            if self.verbose:
                logger.info("📝 Generating director feedback")

            task = f"History: {self.conversation.get_str()} \n\n"

            feedback_director = Agent(
                agent_name="Director",
                agent_description="Director module that provides feedback to the worker agents",
                model_name=self.director_model_name,
                max_loops=1,
                system_prompt=HIEARCHICAL_SWARM_SYSTEM_PROMPT,
            )

            output = feedback_director.run(
                task=(
                    "You are the Director. Carefully review the outputs generated by all the worker agents in the previous step. "
                    "Provide specific, actionable feedback for each agent, highlighting strengths, weaknesses, and concrete suggestions for improvement. "
                    "If any outputs are unclear, incomplete, or could be enhanced, explain exactly how. "
                    f"Your feedback should help the agents refine their work in the next iteration. "
                    f"Worker Agent Responses: {task}"
                )
            )
            self.conversation.add(
                role=self.director.agent_name, content=output
            )

            if self.verbose:
                logger.success(
                    "✅ Director feedback generated successfully"
                )

            return output

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def call_single_agent(
        self, agent_name: str, task: str, *args, **kwargs
    ):
        """
        Call a single agent by name to execute a specific task.

        This method locates an agent by name and executes the given task with
        the current conversation context. The agent's output is added to the
        conversation history for future reference.

        Args:
            agent_name (str): The name of the agent to call.
            task (str): The task to be executed by the agent.
            *args: Additional positional arguments for the agent.
            **kwargs: Additional keyword arguments for the agent.

        Returns:
            Any: The output from the agent's execution.

        Raises:
            ValueError: If the specified agent is not found in the swarm.
            Exception: If agent execution fails.
        """
        try:
            if self.verbose:
                logger.info(f"📞 Calling agent: {agent_name}")

            # Find agent by name
            agent = None
            for a in self.agents:
                if (
                    hasattr(a, "agent_name")
                    and a.agent_name == agent_name
                ):
                    agent = a
                    break

            if agent is None:
                available_agents = [
                    a.agent_name
                    for a in self.agents
                    if hasattr(a, "agent_name")
                ]
                raise ValueError(
                    f"Agent '{agent_name}' not found in swarm. Available agents: {available_agents}"
                )

            output = agent.run(
                task=f"History: {self.conversation.get_str()} \n\n Task: {task}",
                *args,
                **kwargs,
            )
            self.conversation.add(role=agent_name, content=output)

            if self.verbose:
                logger.success(
                    f"✅ Agent {agent_name} completed task successfully"
                )

            return output

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def parse_orders(self, output):
        """
        Parse the director's output to extract plan and orders.

        This method handles various output formats from the director agent and
        extracts the plan and hierarchical orders. It supports both direct
        dictionary formats and function call formats with JSON arguments.

        Args:
            output: The raw output from the director agent.

        Returns:
            tuple: A tuple containing (plan, orders) where plan is a string
                   and orders is a list of HierarchicalOrder objects.

        Raises:
            ValueError: If the output format is unexpected or cannot be parsed.
            Exception: If parsing fails due to other errors.
        """
        try:
            if self.verbose:
                logger.info("📋 Parsing director orders")
                logger.debug(f"📊 Output type: {type(output)}")

            import json

            # Handle different output formats from the director
            if isinstance(output, list):
                # If output is a list, look for function call data
                for item in output:
                    if isinstance(item, dict):
                        # Check if it's a conversation format with role/content
                        if "content" in item and isinstance(
                            item["content"], list
                        ):
                            for content_item in item["content"]:
                                if (
                                    isinstance(content_item, dict)
                                    and "function" in content_item
                                ):
                                    function_data = content_item[
                                        "function"
                                    ]
                                    if "arguments" in function_data:
                                        try:
                                            args = json.loads(
                                                function_data[
                                                    "arguments"
                                                ]
                                            )
                                            if (
                                                "plan" in args
                                                and "orders" in args
                                            ):
                                                plan = args["plan"]
                                                orders = [
                                                    HierarchicalOrder(
                                                        **order
                                                    )
                                                    for order in args[
                                                        "orders"
                                                    ]
                                                ]

                                                if self.verbose:
                                                    logger.success(
                                                        f"✅ Successfully parsed plan and {len(orders)} orders"
                                                    )

                                                return plan, orders
                                        except (
                                            json.JSONDecodeError
                                        ) as json_err:
                                            if self.verbose:
                                                logger.warning(
                                                    f"⚠️ JSON decode error: {json_err}"
                                                )
                                            pass
                        # Check if it's a direct function call format
                        elif "function" in item:
                            function_data = item["function"]
                            if "arguments" in function_data:
                                try:
                                    args = json.loads(
                                        function_data["arguments"]
                                    )
                                    if (
                                        "plan" in args
                                        and "orders" in args
                                    ):
                                        plan = args["plan"]
                                        orders = [
                                            HierarchicalOrder(**order)
                                            for order in args[
                                                "orders"
                                            ]
                                        ]

                                        if self.verbose:
                                            logger.success(
                                                f"✅ Successfully parsed plan and {len(orders)} orders"
                                            )

                                        return plan, orders
                                except (
                                    json.JSONDecodeError
                                ) as json_err:
                                    if self.verbose:
                                        logger.warning(
                                            f"⚠️ JSON decode error: {json_err}"
                                        )
                                    pass
                # If no function call found, raise error
                raise ValueError(
                    f"Unable to parse orders from director output: {output}"
                )
            elif isinstance(output, dict):
                # Handle direct dictionary format
                if "plan" in output and "orders" in output:
                    plan = output["plan"]
                    orders = [
                        HierarchicalOrder(**order)
                        for order in output["orders"]
                    ]

                    if self.verbose:
                        logger.success(
                            f"✅ Successfully parsed plan and {len(orders)} orders"
                        )

                    return plan, orders
                else:
                    raise ValueError(
                        f"Missing 'plan' or 'orders' in director output: {output}"
                    )
            else:
                raise ValueError(
                    f"Unexpected output format from director: {type(output)}"
                )

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def execute_orders(self, orders: list):
        """
        Execute all orders from the director's output.

        This method iterates through all hierarchical orders and calls the
        appropriate agents to execute their assigned tasks. Each agent's
        output is collected and returned as a list.

        Args:
            orders (list): List of HierarchicalOrder objects to execute.

        Returns:
            list: List of outputs from all executed orders.

        Raises:
            Exception: If order execution fails.
        """
        try:
            if self.verbose:
                logger.info(f"⚡ Executing {len(orders)} orders")

            outputs = []
            for i, order in enumerate(orders):
                if self.verbose:
                    logger.info(
                        f"📋 Executing order {i+1}/{len(orders)}: {order.agent_name}"
                    )

                output = self.call_single_agent(
                    order.agent_name, order.task
                )
                outputs.append(output)

            if self.verbose:
                logger.success(
                    f"✅ All {len(orders)} orders executed successfully"
                )

            return outputs

        except Exception as e:
            error_msg = f"❌ Failed to setup director: {str(e)}\n🔍 Traceback: {traceback.format_exc()}\n🐛 If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            logger.error(error_msg)

    def batched_run(
        self, tasks: List[str], img: str = None, *args, **kwargs
    ):
        """
        Execute the hierarchical swarm for multiple tasks in sequence.

        This method processes a list of tasks sequentially, running the complete
        swarm workflow for each task. Each task is processed independently with
        its own conversation context and results.

        Args:
            tasks (List[str]): List of tasks to be processed by the swarm.
            img (str, optional): Optional image input for the tasks.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list: List of results for each processed task.

        Raises:
            Exception: If batched execution fails.
        """
        try:
            if self.verbose:
                logger.info(
                    f"🚀 Starting batched hierarchical swarm run: {self.name}"
                )
                logger.info(
                    f"📊 Configuration - Max loops: {self.max_loops}"
                )

            # Initialize a list to store the results
            results = []

            # Process each task in parallel
            for task in tasks:
                result = self.run(task, img, *args, **kwargs)
                results.append(result)

            if self.verbose:
                logger.success(
                    f"🎉 Batched hierarchical swarm run completed: {self.name}"
                )
                logger.info(f"📊 Total tasks processed: {len(tasks)}")

            return results

        except Exception as e:
            error_msg = (
                f"❌ Batched hierarchical swarm run failed: {str(e)}"
            )
            if self.verbose:
                logger.error(error_msg)
                logger.error(
                    f"🔍 Traceback: {traceback.format_exc()}"
                )
