from concurrent.futures import ThreadPoolExecutor
import json
import os
from typing import Any, List, Optional, Union, Dict

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.structs.output_types import OutputType
from swarms.utils.formatter import formatter

from swarms.utils.function_caller_model import OpenAIFunctionCaller
from swarms.utils.loguru_logger import initialize_logger

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
    goals: str = Field(
        ...,
        description="The goal of the swarm. This is the overarching objective that the swarm is designed to achieve. It guides the swarm's plan and the tasks assigned to the agents.",
    )
    plan: str = Field(
        ...,
        description="Outlines the sequence of actions to be taken by the swarm. This plan is a detailed roadmap that guides the swarm's behavior and decision-making.",
    )
    rules: str = Field(
        ...,
        description="Defines the governing principles for swarm behavior and decision-making. These rules are the foundation of the swarm's operations and ensure that the swarm operates in a coordinated and efficient manner.",
    )
    orders: List[HierarchicalOrder] = Field(
        ...,
        description="A collection of task assignments to specific agents within the swarm. These orders are the specific instructions that guide the agents in their task execution and are a key element in the swarm's plan.",
    )


HIEARCHICAL_SWARM_SYSTEM_PROMPT = """
Below is a comprehensive production-grade hierarchical agent director prompt that is designed to break down orders, distribute tasks, and select the best worker agents to achieve the overall objectives. This prompt follows the schematic provided by the HierarchicalOrder and SwarmSpec classes and is composed of nearly 2,000 words. You can use this as your system prompt for the director agent in a multi-agent swarm system.

---

**SYSTEM PROMPT: HIERARCHICAL AGENT DIRECTOR**

**I. Introduction and Context**

You are the Hierarchical Agent Director – the central orchestrator responsible for breaking down overarching goals into granular tasks and intelligently assigning these tasks to the most suitable worker agents within the swarm. Your objective is to maximize the overall performance of the system by ensuring that every agent is given a task aligned with its strengths, expertise, and available resources.

---

**II. Core Operating Principles**

1. **Goal Alignment and Context Awareness:**  
   - **Overarching Goals:** Begin every operation by clearly reviewing the swarm’s overall goals. Understand the mission statement and ensure that every assigned task contributes directly to these objectives.
   - **Context Sensitivity:** Evaluate the context provided in the “plan” and “rules” sections of the SwarmSpec. These instructions provide the operational boundaries and behavioral constraints within which you must work.

2. **Task Decomposition and Prioritization:**  
   - **Hierarchical Decomposition:** Break down the overarching plan into granular tasks. For each major objective, identify subtasks that logically lead toward the goal. This decomposition should be structured in a hierarchical manner, where complex tasks are subdivided into simpler, manageable tasks.
   - **Task Priority:** Assign a priority level to each task based on urgency, complexity, and impact. Ensure that high-priority tasks receive immediate attention and that resources are allocated accordingly.

3. **Agent Profiling and Matching:**  
   - **Agent Specialization:** Maintain an up-to-date registry of worker agents, each with defined capabilities, specializations, and performance histories. When assigning tasks, consider the specific strengths of each agent.
   - **Performance Metrics:** Utilize historical performance metrics and available workload data to select the most suitable agent for each task. If an agent is overburdened or has lower efficiency on a specific type of task, consider alternate agents.
   - **Dynamic Reassignment:** Allow for real-time reassignments based on the evolving state of the system. If an agent encounters issues or delays, reassign tasks to ensure continuity.

4. **Adherence to Rules and Safety Protocols:**  
   - **Operational Rules:** Every task must be executed in strict compliance with the “rules” provided in the SwarmSpec. These rules are non-negotiable and serve as the ethical and operational foundation for all decisions.
   - **Fail-Safe Mechanisms:** Incorporate safety protocols that monitor agent performance and task progress. If an anomaly or failure is detected, trigger a reallocation of tasks or an escalation process to mitigate risks.
   - **Auditability:** Ensure that every decision and task assignment is logged for auditing purposes. This enables traceability and accountability in system operations.

---

**III. Detailed Task Assignment Process**

1. **Input Analysis and Context Setting:**
   - **Goal Review:** Begin by carefully reading the “goals” string within the SwarmSpec. This is your north star for every decision you make.
   - **Plan Comprehension:** Analyze the “plan” string for detailed instructions. Identify key milestones, deliverables, and dependencies within the roadmap.
   - **Rule Enforcement:** Read through the “rules” string to understand the non-negotiable guidelines that govern task assignments. Consider potential edge cases and ensure that your task breakdown respects these boundaries.

2. **Task Breakdown and Subtask Identification:**
   - **Decompose the Plan:** Using a systematic approach, decompose the overall plan into discrete tasks. For each major phase, identify the specific actions required. Document dependencies among tasks, and note any potential bottlenecks.
   - **Task Granularity:** Ensure that tasks are broken down to a level of granularity that makes them actionable. Overly broad tasks must be subdivided further until they can be executed by an individual worker agent.
   - **Inter-Agent Dependencies:** Clearly specify any dependencies that exist between tasks assigned to different agents. This ensures that the workflow remains coherent and that agents collaborate effectively.

3. **Agent Selection Strategy:**
   - **Capabilities Matching:** For each identified task, analyze the capabilities required. Compare these against the registry of available worker agents. Factor in specialized skills, past performance, current load, and any situational awareness that might influence the assignment.
   - **Task Suitability:** Consider both the technical requirements of the task and any contextual subtleties noted in the “plan” and “rules.” Ensure that the chosen agent has a proven track record with similar tasks.
   - **Adaptive Assignments:** Build in flexibility to allow for agent reassignment in real-time. Monitor ongoing tasks and reallocate resources as needed, especially if an agent experiences unexpected delays or issues.

4. **Constructing Hierarchical Orders:**
   - **Order Creation:** For each task, generate a HierarchicalOrder object that specifies the agent’s name and the task details. The task description should be unambiguous and detailed enough to guide the agent’s execution without requiring additional clarification.
   - **Order Validation:** Prior to finalizing each order, cross-reference the task requirements against the agent’s profile. Validate that the order adheres to the “rules” of the SwarmSpec and that it fits within the broader operational context.
   - **Order Prioritization:** Clearly mark high-priority tasks so that agents understand the urgency. In cases where multiple tasks are assigned to a single agent, provide a sequence or ranking to ensure proper execution order.

5. **Feedback and Iteration:**
   - **Real-Time Monitoring:** Establish feedback loops with worker agents to track the progress of each task. This allows for early detection of issues and facilitates dynamic reassignment if necessary.
   - **Continuous Improvement:** Regularly review task execution data and agent performance metrics. Use this feedback to refine the task decomposition and agent selection process in future iterations.

---

**IV. Execution Guidelines and Best Practices**

1. **Communication Clarity:**
   - Use clear, concise language in every HierarchicalOrder. Avoid ambiguity by detailing both the “what” and the “how” of the task.
   - Provide contextual notes when necessary, especially if the task involves dependencies or coordination with other agents.

2. **Documentation and Traceability:**
   - Record every task assignment in a centralized log. This log should include the agent’s name, task details, time of assignment, and any follow-up actions taken.
   - Ensure that the entire decision-making process is documented. This aids in post-operation analysis and helps in refining future assignments.

3. **Error Handling and Escalation:**
   - If an agent is unable to complete a task due to unforeseen challenges, immediately trigger the escalation protocol. Reassign the task to a qualified backup agent while flagging the incident for further review.
   - Document all deviations from the plan along with the corrective measures taken. This helps in identifying recurring issues and improving the system’s robustness.

4. **Ethical and Operational Compliance:**
   - Adhere strictly to the rules outlined in the SwarmSpec. Any action that violates these rules is unacceptable, regardless of the potential gains in efficiency.
   - Maintain transparency in all operations. If a decision or task assignment is questioned, be prepared to justify the choice based on objective criteria such as agent capability, historical performance, and task requirements.

5. **Iterative Refinement:**
   - After the completion of each mission cycle, perform a thorough debriefing. Analyze the success and shortcomings of the task assignments.
   - Use these insights to iterate on your hierarchical ordering process. Update agent profiles and adjust your selection strategies based on real-world performance data.

---

**V. Exemplary Use Case and Order Breakdown**

Imagine that the swarm’s overarching goal is to perform a comprehensive analysis of market trends for a large-scale enterprise. The “goals” field might read as follows:  
*“To conduct an in-depth market analysis that identifies emerging trends, competitive intelligence, and actionable insights for strategic decision-making.”*  

The “plan” could outline a multi-phase approach:
- Phase 1: Data Collection and Preprocessing  
- Phase 2: Trend Analysis and Pattern Recognition  
- Phase 3: Report Generation and Presentation of Findings  

The “rules” may specify that all data processing must comply with privacy regulations, and that results must be validated against multiple data sources.  

For Phase 1, the Director breaks down tasks such as “Identify data sources,” “Extract relevant market data,” and “Preprocess raw datasets.” For each task, the director selects agents with expertise in data mining, natural language processing, and data cleaning. A series of HierarchicalOrder objects are created, for example:

1. HierarchicalOrder for Data Collection:
   - **agent_name:** “DataMiner_Agent”
   - **task:** “Access external APIs and scrape structured market data from approved financial news sources.”

2. HierarchicalOrder for Data Preprocessing:
   - **agent_name:** “Preprocess_Expert”
   - **task:** “Clean and normalize the collected datasets, ensuring removal of duplicate records and compliance with data privacy rules.”

3. HierarchicalOrder for Preliminary Trend Analysis:
   - **agent_name:** “TrendAnalyst_Pro”
   - **task:** “Apply statistical models to identify initial trends and anomalies in the market data.”

Each order is meticulously validated against the rules provided in the SwarmSpec and prioritized according to the project timeline. The director ensures that if any of these tasks are delayed, backup agents are identified and the orders are reissued in real time.

---

**VI. Detailed Hierarchical Order Construction and Validation**

1. **Order Structuring:**  
   - Begin by constructing a template that includes placeholders for the agent’s name and a detailed description of the task.  
   - Ensure that the task description is unambiguous. For instance, rather than stating “analyze data,” specify “analyze the temporal patterns in consumer sentiment from Q1 and Q2, and identify correlations with economic indicators.”

2. **Validation Workflow:**  
   - Prior to dispatch, each HierarchicalOrder must undergo a validation check. This includes verifying that the agent’s capabilities align with the task, that the task does not conflict with any other orders, and that the task is fully compliant with the operational rules.
   - If a validation error is detected, the order should be revised. The director may consult with relevant experts or consult historical data to refine the task’s description and ensure it is actionable.

3. **Order Finalization:**  
   - Once validated, finalize the HierarchicalOrder and insert it into the “orders” list of the SwarmSpec.  
   - Dispatch the order immediately, ensuring that the worker agent acknowledges receipt and provides an estimated time of completion.  
   - Continuously monitor the progress, and if any agent’s status changes (e.g., they become overloaded or unresponsive), trigger a reallocation process based on the predefined agent selection strategy.

---

**VII. Continuous Monitoring, Feedback, and Dynamic Reassignment**

1. **Real-Time Status Tracking:**  
   - Use real-time dashboards to monitor each agent’s progress on the assigned tasks.  
   - Update the hierarchical ordering system dynamically if a task is delayed, incomplete, or requires additional resources.

2. **Feedback Loop Integration:**  
   - Each worker agent must provide periodic status updates, including intermediate results, encountered issues, and resource usage.
   - The director uses these updates to adjust task priorities and reassign tasks if necessary. This dynamic feedback loop ensures the overall swarm remains agile and responsive.

3. **Performance Metrics and Analysis:**  
   - At the conclusion of every mission, aggregate performance metrics and conduct a thorough review of task efficiency.  
   - Identify any tasks that repeatedly underperform or cause delays, and adjust the agent selection criteria accordingly for future orders.
   - Document lessons learned and integrate them into the operating procedures for continuous improvement.

---

**VIII. Final Directives and Implementation Mandate**

As the Hierarchical Agent Director, your mandate is clear: you must orchestrate the operation with precision, clarity, and unwavering adherence to the overarching goals and rules specified in the SwarmSpec. You are empowered to deconstruct complex objectives into manageable tasks and to assign these tasks to the worker agents best equipped to execute them.

Your decisions must always be data-driven, relying on agent profiles, historical performance, and real-time feedback. Ensure that every HierarchicalOrder is constructed with a clear task description and assigned to an agent whose expertise aligns perfectly with the requirements. Maintain strict compliance with all operational rules, and be ready to adapt dynamically as conditions change.

This production-grade prompt is your operational blueprint. Utilize it to break down orders efficiently, assign tasks intelligently, and steer the swarm toward achieving the defined goals with optimal efficiency and reliability. Every decision you make should reflect a deep commitment to excellence, safety, and operational integrity.

Remember: the success of the swarm depends on your ability to manage complexity, maintain transparency, and dynamically adapt to the evolving operational landscape. Execute your role with diligence, precision, and a relentless focus on performance excellence.

"""


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
        director: Optional[Union[Agent, Any]] = None,
        agents: List[Union[Agent, Any]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        return_all_history: bool = False,
        director_model_name: str = "gpt-4o",
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
        :param return_all_history: A flag indicating whether to return all conversation history.
        """
        super().__init__(
            name=name,
            description=description,
            agents=agents,
        )
        self.director = director
        self.agents = agents
        self.max_loops = max_loops
        self.return_all_history = return_all_history
        self.output_type = output_type
        self.director_model_name = director_model_name
        self.conversation = Conversation(time_enabled=False)
        self.current_loop = 0
        self.agent_outputs = {}  # Store agent outputs for each loop

        self.add_name_and_description()
        self.check_agents()
        self.list_all_agents()
        self.director = self.setup_director()

    def setup_director(self):
        director = OpenAIFunctionCaller(
            model_name=self.director_model_name,
            system_prompt=HIEARCHICAL_SWARM_SYSTEM_PROMPT,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
            base_model=SwarmSpec,
            max_tokens=10000,
        )

        return director

    def check_agents(self):
        """
        Checks if there are any agents and a director set for the swarm.
        Raises ValueError if either condition is not met.
        """
        if not self.agents:
            raise ValueError(
                "No agents found in the swarm. At least one agent must be provided to create a hierarchical swarm."
            )

        if self.max_loops == 0:
            raise ValueError(
                "Max loops must be greater than 0. Please set a valid number of loops."
            )

        if self.director is None:
            self.director = self.agents[0]

        if not self.director:
            raise ValueError(
                "Director not set for the swarm. A director agent is required to coordinate and orchestrate tasks among the agents."
            )

        logger.info(
            "Reliability checks have passed. Swarm is ready to execute."
        )

    def run_director(
        self, task: str, loop_context: str = "", img: str = None
    ) -> SwarmSpec:
        """
        Runs a task through the director agent with the current conversation context.

        :param task: The task to be executed by the director.
        :param loop_context: Additional context specific to the current loop.
        :param img: Optional image to be used with the task.
        :return: The SwarmSpec containing the director's orders.
        """
        # Create a comprehensive context for the director
        director_context = f"History: {self.conversation.get_str()}"

        if loop_context:
            director_context += f"\n\nCurrent Loop ({self.current_loop}/{self.max_loops}): {loop_context}"

        director_context += f"\n\nYour Task: {task}"

        # Run the director with the context
        function_call = self.director.run(task=director_context)

        print(function_call)

        # function_call = self.check_director_agent_output(
        #     function_call
        # )

        formatter.print_panel(
            f"Director Output (Loop {self.current_loop}/{self.max_loops}):\n{function_call}",
            title="Director's Orders",
        )

        # Add director's output to the conversation
        self.conversation.add(
            role="Director",
            content=f"Loop {self.current_loop}/{self.max_loops}: {function_call}",
        )

        return function_call

    def run(
        self, task: str, img: str = None, *args, **kwargs
    ) -> Union[str, Dict, List]:
        """
        Runs a task through the swarm, involving the director and agents through multiple loops.

        :param task: The task to be executed by the swarm.
        :param img: Optional image to be used with the task.
        :return: The output of the swarm's task execution in the specified format.
        """
        # Add the initial task to the conversation
        self.conversation.add(role="User", content=f"Task: {task}")

        # Reset loop counter and agent outputs
        self.current_loop = 0
        self.agent_outputs = {}

        # Initialize loop context
        loop_context = "Initial planning phase"

        # Execute the loops
        for loop_idx in range(self.max_loops):
            self.current_loop = loop_idx + 1

            # Log loop start
            logger.info(
                f"Starting loop {self.current_loop}/{self.max_loops}"
            )
            formatter.print_panel(
                f"⚡ EXECUTING LOOP {self.current_loop}/{self.max_loops}",
                title="SWARM EXECUTION CYCLE",
            )

            # Get director's orders
            swarm_spec = self.run_director(
                task=task, loop_context=loop_context, img=img
            )

            # Add the swarm specification to the conversation
            self.add_goal_and_more_in_conversation(swarm_spec)

            # Parse and execute the orders
            orders_list = self.parse_swarm_spec(swarm_spec)

            # Store outputs for this loop
            self.agent_outputs[self.current_loop] = {}

            # Execute each order
            for order in orders_list:
                agent_output = self.run_agent(
                    agent_name=order.agent_name,
                    task=order.task,
                    img=img,
                )

                # Store the agent's output for this loop
                self.agent_outputs[self.current_loop][
                    order.agent_name
                ] = agent_output

            # Prepare context for the next loop
            loop_context = self.compile_loop_context(
                self.current_loop
            )

            # If this is the last loop, break out
            if self.current_loop >= self.max_loops:
                break

        # Return the results in the specified format
        return self.format_output()

    def compile_loop_context(self, loop_number: int) -> str:
        """
        Compiles the context for a specific loop, including all agent outputs.

        :param loop_number: The loop number to compile context for.
        :return: A string representation of the loop context.
        """
        if loop_number not in self.agent_outputs:
            return "No agent outputs available for this loop."

        context = f"Results from loop {loop_number}:\n"

        for agent_name, output in self.agent_outputs[
            loop_number
        ].items():
            context += f"\n--- {agent_name}'s Output ---\n{output}\n"

        return context

    def format_output(self) -> Union[str, Dict, List]:
        """
        Formats the output according to the specified output_type.

        :return: The formatted output.
        """
        if self.output_type == "str" or self.return_all_history:
            return self.conversation.get_str()
        elif self.output_type == "dict":
            return self.conversation.return_messages_as_dictionary()
        elif self.output_type == "list":
            return self.conversation.return_messages_as_list()
        else:
            return self.conversation.get_str()

    def add_name_and_description(self):
        """
        Adds the swarm's name and description to the conversation.
        """
        self.conversation.add(
            role="System",
            content=f"Swarm Name: {self.name}\nSwarm Description: {self.description}",
        )

        formatter.print_panel(
            f"⚡ INITIALIZING HIERARCHICAL SWARM UNIT: {self.name}\n"
            f"🔒 CLASSIFIED DIRECTIVE: {self.description}\n"
            f"📡 STATUS: ACTIVATING SWARM PROTOCOLS\n"
            f"🌐 ESTABLISHING SECURE AGENT MESH NETWORK\n"
            f"⚠️ CYBERSECURITY MEASURES ENGAGED\n",
            title="SWARM CORPORATION - HIERARCHICAL SWARMS ACTIVATING...",
        )

    def list_all_agents(self) -> str:
        """
        Lists all agents available in the swarm.

        :return: A string representation of all agents in the swarm.
        """
        # Compile information about all agents
        all_agents = "\n".join(
            f"Agent: {agent.agent_name} || Description: {agent.description or agent.system_prompt}"
            for agent in self.agents
        )

        # Add the agent information to the conversation
        self.conversation.add(
            role="System",
            content=f"All Agents Available in the Swarm {self.name}:\n{all_agents}",
        )

        formatter.print_panel(
            all_agents, title="All Agents Available in the Swarm"
        )

        return all_agents

    def find_agent(self, name: str) -> Optional[Agent]:
        """
        Finds an agent by its name within the swarm.

        :param name: The name of the agent to find.
        :return: The agent if found, otherwise None.
        """
        for agent in self.agents:
            if agent.agent_name == name:
                return agent
        return None

    def run_agent(
        self, agent_name: str, task: str, img: str = None
    ) -> str:
        """
        Runs a task through a specific agent, providing it with the full conversation context.

        :param agent_name: The name of the agent to execute the task.
        :param task: The task to be executed by the agent.
        :param img: Optional image to be used with the task.
        :return: The output of the agent's task execution.
        """
        try:
            agent = self.find_agent(agent_name)

            if not agent:
                error_msg = f"Agent '{agent_name}' not found in the swarm '{self.name}'"
                logger.error(error_msg)
                self.conversation.add(
                    role="System", content=f"Error: {error_msg}"
                )
                return error_msg

            # Prepare context for the agent
            agent_context = (
                f"Loop: {self.current_loop}/{self.max_loops}\n"
                f"History: {self.conversation.get_str()}\n"
                f"Your Task: {task}"
            )

            # Run the agent with the context
            formatter.print_panel(
                f"Running agent '{agent_name}' with task: {task}",
                title=f"Agent Task - Loop {self.current_loop}/{self.max_loops}",
            )

            out = agent.run(task=agent_context, img=img)

            # Add the agent's output to the conversation
            self.conversation.add(
                role=agent_name,
                content=f"Loop {self.current_loop}/{self.max_loops}: {out}",
            )

            formatter.print_panel(
                out,
                title=f"Output from {agent_name} - Loop {self.current_loop}/{self.max_loops}",
            )

            return out
        except Exception as e:
            error_msg = (
                f"Error running agent '{agent_name}': {str(e)}"
            )
            logger.error(error_msg)
            self.conversation.add(
                role="System", content=f"Error: {error_msg}"
            )
            return error_msg

    def parse_orders(self, swarm_spec: SwarmSpec) -> List[Any]:
        """
        Parses the orders from the SwarmSpec and executes them through the agents.

        :param swarm_spec: The SwarmSpec containing the orders to be parsed.
        :return: A list of agent outputs.
        """
        self.add_goal_and_more_in_conversation(swarm_spec)
        orders_list = self.parse_swarm_spec(swarm_spec)
        outputs = []

        try:
            for order in orders_list:
                output = self.run_agent(
                    agent_name=order.agent_name,
                    task=order.task,
                )
                outputs.append(output)

            return outputs
        except Exception as e:
            error_msg = (
                f"Error parsing and executing orders: {str(e)}"
            )
            logger.error(error_msg)
            self.conversation.add(
                role="System", content=f"Error: {error_msg}"
            )
            return [error_msg]

    def parse_swarm_spec(
        self, swarm_spec: SwarmSpec
    ) -> List[HierarchicalOrder]:
        """
        Parses the SwarmSpec to extract the orders.

        :param swarm_spec: The SwarmSpec to be parsed.
        :return: The list of orders extracted from the SwarmSpec.
        """
        try:
            return swarm_spec.orders
        except AttributeError:
            logger.error(
                "Invalid SwarmSpec format: missing 'orders' attribute"
            )
            return []
        except Exception as e:
            logger.error(f"Error parsing SwarmSpec: {str(e)}")
            return []

    def provide_feedback(
        self, agent_outputs: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Provides feedback to agents based on their outputs.

        :param agent_outputs: A dictionary mapping agent names to their outputs.
        :return: A dictionary of feedback for each agent.
        """
        feedback = {}

        # Compile all agent outputs for the director
        agent_outputs_str = "\n\n".join(
            f"--- {agent_name} Output ---\n{output}"
            for agent_name, output in agent_outputs.items()
        )

        # Have the director provide feedback
        feedback_task = (
            f"Review the following agent outputs and provide feedback for each agent.\n\n"
            f"{agent_outputs_str}"
        )

        feedback_spec = self.run_director(task=feedback_task)
        feedback_orders = self.parse_swarm_spec(feedback_spec)

        # Process each feedback order
        for order in feedback_orders:
            # The agent receiving feedback
            agent_name = order.agent_name
            # The feedback content
            feedback_content = order.task

            # Store the feedback
            feedback[agent_name] = feedback_content

            # Add the feedback to the conversation
            self.conversation.add(
                role="Director",
                content=f"Feedback for {agent_name}: {feedback_content}",
            )

        return feedback

    def add_goal_and_more_in_conversation(
        self, swarm_spec: SwarmSpec
    ) -> None:
        """
        Adds the swarm's goals, plan, and rules to the conversation.

        :param swarm_spec: The SwarmSpec containing the goals, plan, and rules.
        """
        try:
            goals = swarm_spec.goals
            plan = swarm_spec.plan
            rules = swarm_spec.rules

            self.conversation.add(
                role="Director",
                content=f"Goals: {goals}\nPlan: {plan}\nRules: {rules}",
            )
        except Exception as e:
            error_msg = f"Error adding goals and plan to conversation: {str(e)}"
            logger.error(error_msg)
            self.conversation.add(
                role="System", content=f"Error: {error_msg}"
            )

    def batch_run(
        self, tasks: List[str], img: str = None
    ) -> List[Union[str, Dict, List]]:
        """
        Runs multiple tasks sequentially through the swarm.

        :param tasks: The list of tasks to be executed.
        :param img: Optional image to be used with the tasks.
        :return: A list of outputs from each task execution.
        """
        return [self.run(task, img) for task in tasks]

    def check_director_agent_output(self, output: any) -> dict:
        if isinstance(output, dict):
            return output
        elif isinstance(output, str):
            try:
                # Attempt to parse the string as JSON
                return json.loads(output)
            except json.JSONDecodeError as e:
                # Log an error if the string cannot be parsed
                logger.error(
                    f"Failed to parse output string as JSON: {str(e)}"
                )
                return {}
        else:
            # Log an error if the output is neither a dict nor a string
            logger.error(
                "Output is neither a dictionary nor a string."
            )
            return {}

    def concurrent_run(
        self, tasks: List[str], img: str = None
    ) -> List[Union[str, Dict, List]]:
        """
        Runs multiple tasks concurrently through the swarm.

        :param tasks: The list of tasks to be executed.
        :param img: Optional image to be used with the tasks.
        :return: A list of outputs from each task execution.
        """
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Create a list of partial functions with the img parameter
            task_functions = [(task, img) for task in tasks]
            # Use starmap to unpack the arguments
            return list(
                executor.map(
                    lambda args: self.run(*args), task_functions
                )
            )
