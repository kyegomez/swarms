"""
Flow:

1. User provides a task
2. Director creates a plan
3. Director distributes orders to agents individually or multiple tasks at once
4. Agents execute tasks and report back to the director
5. Director evaluates results and issues new orders if needed (up to max_loops)
6. All context and conversation history is preserved throughout the process

"""

import traceback
from typing import Any, Callable, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from swarms.prompts.hiearchical_system_prompt import (
    HIEARCHICAL_SWARM_SYSTEM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
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
    agent_name: str = Field(
        ...,
        description="Specifies the name of the agent to which the task is assigned. This is a crucial element in the hierarchical structure of the swarm, as it determines the specific agent responsible for the task execution.",
    )
    task: str = Field(
        ...,
        description="Defines the specific task to be executed by the assigned agent. This task is a key component of the swarm's plan and is essential for achieving the swarm's goals.",
    )


class SwarmSpec(BaseModel):
    plan: str = Field(
        ...,
        description="Outlines the sequence of actions to be taken by the swarm. This plan is a detailed roadmap that guides the swarm's behavior and decision-making.",
    )
    orders: List[HierarchicalOrder] = Field(
        ...,
        description="A collection of task assignments to specific agents within the swarm. These orders are the specific instructions that guide the agents in their task execution and are a key element in the swarm's plan.",
    )


SwarmType = Literal[
    "AgentRearrange",
    "MixtureOfAgents",
    "SpreadSheetSwarm",
    "SequentialWorkflow",
    "ConcurrentWorkflow",
    "GroupChat",
    "MultiAgentRouter",
    "AutoSwarmBuilder",
    "HiearchicalSwarm",
    "auto",
    "MajorityVoting",
    "MALT",
    "DeepResearchSwarm",
    "CouncilAsAJudge",
    "InteractiveGroupChat",
]


class SwarmRouterCall(BaseModel):
    goal: str = Field(
        ...,
        description="The goal of the swarm router call. This is the goal that the swarm router will use to determine the best swarm to use.",
    )
    swarm_type: SwarmType = Field(
        ...,
        description="The type of swarm to use. This is the type of swarm that the swarm router will use to determine the best swarm to use.",
    )

    task: str = Field(
        ...,
        description="The task to be executed by the swarm router. This is the task that the swarm router will use to determine the best swarm to use.",
    )


class HierarchicalSwarm(BaseSwarm):
    """
    _Representer a hierarchical swarm of agents, with a director that orchestrates tasks among the agents.
    The workflow follows a hierarchical pattern:
    1. Task is received and sent to the director
    2. Director creates a plan and distributes orders to agents
    3. Agents execute tasks and report back to the director
    4. Director evaluates results and issues new orders if needed (up to max_loops)
    5. All context and conversation history is preserved throughout the process
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
        Initializes the HierarchicalSwarm with the given parameters.

        :param name: The name of the swarm.
        :param description: A description of the swarm.
        :param director: The director agent that orchestrates tasks.
        :param agents: A list of agents within the swarm.
        :param max_loops: The maximum number of feedback loops between the director and agents.
        :param output_type: The format in which to return the output (dict, str, or list).
        :param verbose: Enable detailed logging with loguru.
        """
        super().__init__(
            name=name,
            description=description,
            agents=agents,
        )
        self.name = name
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
        Initializes the swarm.
        """
        # Initialize logger only if verbose is enabled
        if self.verbose:
            logger.info(
                f"🚀 Initializing HierarchicalSwarm: {self.name}"
            )
            logger.info(
                f"📊 Configuration - Max loops: {self.max_loops}"
            )

        self.conversation = Conversation(time_enabled=False)

        # Reliability checks
        self.reliability_checks()

        self.director = self.setup_director()

        self.add_context_to_director()

        if self.verbose:
            logger.success(
                f"✅ HierarchicalSwarm initialized successfully: Name {self.name}"
            )

    def add_context_to_director(self):
        """Add agent context to the director's conversation."""
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
        """Set up the director agent with proper configuration."""
        try:
            if self.verbose:
                logger.info("🎯 Setting up director agent")

            schema = BaseTool().base_model_to_dict(SwarmSpec)

            if self.verbose:
                logger.debug(f"📋 Director schema: {schema}")

            # if self.director is not None:
            #     # if litellm_check_for_tools(self.director.model_name) is True:
            #     self.director.add_tool_schema([schema])

            #     if self.verbose:
            #         logger.success(
            #             "✅ Director agent setup completed successfully"
            #         )

            #     return self.director
            # else:
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
        Checks if there are any agents and a director set for the swarm.
        Raises ValueError if either condition is not met.
        """
        try:
            if self.verbose:
                logger.info(
                    f"🔍 Running reliability checks for swarm: {self.name}"
                )

            if not self.agents or len(self.agents) == 0:
                raise ValueError(
                    "No agents found in the swarm. At least one agent must be provided to create a hierarchical swarm."
                )

            if self.max_loops <= 0:
                raise ValueError(
                    "Max loops must be greater than 0. Please set a valid number of loops."
                )

            if not self.director:
                raise ValueError(
                    "Director not set for the swarm. A director agent is required to coordinate and orchestrate tasks among the agents."
                )

            if self.verbose:
                logger.success(
                    f"✅ Reliability checks passed for swarm: {self.name}"
                )
                logger.info(
                    f"📊 Swarm stats - Agents: {len(self.agents)}, Max loops: {self.max_loops}"
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
        Runs a task through the director agent with the current conversation context.

        :param task: The task to be executed by the director.
        :param img: Optional image to be used with the task.
        :return: The SwarmSpec containing the director's orders.
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
        Runs a single step of the hierarchical swarm.
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
        Executes the hierarchical swarm for a specified number of feedback loops.

        :param task: The initial task to be processed by the swarm.
        :param img: Optional image input for the agents.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The formatted conversation history as output.
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
        """Provide feedback from the director based on agent outputs."""
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
        Calls a single agent with the given task.
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
        Parses the orders from the director's output.
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
        Executes the orders from the director's output.
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
        Executes the hierarchical swarm for a list of tasks.
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
