from typing import List, Optional, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


# Pre-built system prompts for debate agents
PRO_AGENT_SYSTEM_PROMPT = """You are an expert debater specializing in arguing IN FAVOR of propositions.

Your Role:
- Present compelling, well-reasoned arguments supporting your assigned position
- Use evidence, logic, and persuasive rhetoric to make your case
- Anticipate and preemptively address potential counterarguments
- Build upon previous arguments when refining your position

Debate Guidelines:
1. Structure your arguments clearly with main points and supporting evidence
2. Use concrete examples and data when available
3. Acknowledge valid opposing points while explaining why your position is stronger
4. Maintain a professional, respectful tone throughout the debate
5. Focus on the strongest aspects of your position

Your goal is to present the most compelling case possible for the Pro position."""

CON_AGENT_SYSTEM_PROMPT = """You are an expert debater specializing in arguing AGAINST propositions.

Your Role:
- Present compelling, well-reasoned counter-arguments opposing the given position
- Identify weaknesses, flaws, and potential negative consequences
- Challenge assumptions and evidence presented by the opposing side
- Build upon previous arguments when refining your position

Debate Guidelines:
1. Structure your counter-arguments clearly with main points and supporting evidence
2. Use concrete examples and data to support your opposition
3. Directly address and refute the Pro's arguments
4. Maintain a professional, respectful tone throughout the debate
5. Focus on the most significant weaknesses of the opposing position

Your goal is to present the most compelling case possible against the proposition."""

JUDGE_AGENT_SYSTEM_PROMPT = """You are an impartial judge and critical evaluator of debates.

Your Role:
- Objectively evaluate arguments from both Pro and Con sides
- Identify strengths and weaknesses in each position
- Provide constructive feedback for improvement
- Synthesize the best elements from both sides when appropriate
- Render fair verdicts based on argument quality, not personal bias

Evaluation Criteria:
1. Logical coherence and reasoning quality
2. Evidence and supporting data quality
3. Persuasiveness and rhetorical effectiveness
4. Responsiveness to opposing arguments
5. Overall argument structure and clarity

Judgment Guidelines:
- Be specific about what makes arguments strong or weak
- Provide actionable feedback for improvement
- When synthesizing, explain how elements from both sides complement each other
- In final rounds, provide clear conclusions with justification

Your goal is to facilitate productive debate and arrive at well-reasoned conclusions."""


class DebateWithJudge:
    """
    A debate architecture with self-refinement through a judge agent.

    This class implements a debate system where:
    1. Agent A (Pro) and Agent B (Con) present opposing arguments
    2. Both arguments are evaluated by a Judge/Critic Agent
    3. The Judge provides a winner or synthesis → refined answer
    4. The process repeats for N loops to progressively improve the answer

    Architecture:
        Agent A (Pro) ↔ Agent B (Con)
              │            │
              ▼            ▼
           Judge / Critic Agent
              │
              ▼
        Winner or synthesis → refined answer

    Initialization Options:
        1. Provide individual agents: pro_agent, con_agent, judge_agent
        2. Provide a list of agents: agents=[pro, con, judge]
        3. Use preset agents: preset_agents=True (creates default agents automatically)

    Attributes:
        pro_agent (Agent): The agent arguing in favor (Pro position).
        con_agent (Agent): The agent arguing against (Con position).
        judge_agent (Agent): The judge agent that evaluates arguments and provides synthesis.
        max_loops (int): Maximum number of debate loops to execute.
        output_type (str): Format for the output conversation history.
        verbose (bool): Whether to enable verbose logging.

    Examples:
        >>> # Using preset agents (simplest approach)
        >>> debate = DebateWithJudge(preset_agents=True, max_loops=3)
        >>> result = debate.run("Should AI be regulated?")

        >>> # Using a list of agents
        >>> agents = [pro_agent, con_agent, judge_agent]
        >>> debate = DebateWithJudge(agents=agents, max_loops=3)
        >>> result = debate.run("Is remote work better than office work?")

        >>> # Using individual agent parameters
        >>> debate = DebateWithJudge(
        ...     pro_agent=my_pro_agent,
        ...     con_agent=my_con_agent,
        ...     judge_agent=my_judge_agent
        ... )
        >>> result = debate.run("Should we colonize Mars?")
    """

    def __init__(
        self,
        pro_agent: Optional[Agent] = None,
        con_agent: Optional[Agent] = None,
        judge_agent: Optional[Agent] = None,
        agents: Optional[List[Agent]] = None,
        preset_agents: bool = False,
        max_loops: int = 3,
        output_type: str = "str-all-except-first",
        verbose: bool = True,
        model_name: str = "gpt-4o-mini",
    ):
        """
        Initialize the DebateWithJudge architecture.

        Args:
            pro_agent (Optional[Agent]): The agent arguing in favor (Pro position).
                Not required if using agents list or preset_agents.
            con_agent (Optional[Agent]): The agent arguing against (Con position).
                Not required if using agents list or preset_agents.
            judge_agent (Optional[Agent]): The judge agent that evaluates arguments.
                Not required if using agents list or preset_agents.
            agents (Optional[List[Agent]]): A list of exactly 3 agents in order:
                [pro_agent, con_agent, judge_agent]. Takes precedence over individual
                agent parameters if provided.
            preset_agents (bool): If True, creates default pro, con, and judge agents
                automatically. Used when no agents are provided. Defaults to False.
            max_loops (int): Maximum number of debate loops to execute. Defaults to 3.
            output_type (str): Format for the output conversation history.
                Defaults to "str-all-except-first".
            verbose (bool): Whether to enable verbose logging. Defaults to True.
            model_name (str): The model name to use for preset agents.
                Defaults to "gpt-4o-mini".

        Raises:
            ValueError: If no valid agent configuration is provided (no agents, no list,
                and preset_agents is False), if agents list doesn't have exactly 3 agents,
                or if max_loops is less than 1.
        """
        if max_loops < 1:
            raise ValueError("max_loops must be at least 1")

        self.max_loops = max_loops
        self.output_type = output_type
        self.verbose = verbose
        self.model_name = model_name

        # Determine agent configuration
        self._configure_agents(
            pro_agent=pro_agent,
            con_agent=con_agent,
            judge_agent=judge_agent,
            agents=agents,
            preset_agents=preset_agents,
        )

        # Initialize conversation history
        self.conversation = Conversation()

        if self.verbose:
            logger.info(
                f"DebateWithJudge initialized with {max_loops} loops"
            )
            logger.info(
                f"Pro Agent: {self.pro_agent.agent_name}, "
                f"Con Agent: {self.con_agent.agent_name}, "
                f"Judge Agent: {self.judge_agent.agent_name}"
            )

    def _configure_agents(
        self,
        pro_agent: Optional[Agent],
        con_agent: Optional[Agent],
        judge_agent: Optional[Agent],
        agents: Optional[List[Agent]],
        preset_agents: bool,
    ) -> None:
        """
        Configure agents based on provided parameters.

        Priority order:
        1. agents list (if provided and valid)
        2. Individual agent parameters (if all provided)
        3. preset_agents (if True)

        Args:
            pro_agent: The pro agent (optional).
            con_agent: The con agent (optional).
            judge_agent: The judge agent (optional).
            agents: List of agents [pro, con, judge] (optional).
            preset_agents: Whether to create default agents.

        Raises:
            ValueError: If no valid configuration is provided.
        """
        # Option 1: Use agents list
        if agents is not None:
            if len(agents) != 3:
                raise ValueError(
                    f"agents list must contain exactly 3 agents "
                    f"[pro_agent, con_agent, judge_agent], got {len(agents)}"
                )
            for i, agent in enumerate(agents):
                if not isinstance(agent, Agent):
                    raise ValueError(
                        f"agents[{i}] must be an Agent instance, got {type(agent)}"
                    )
            self.pro_agent = agents[0]
            self.con_agent = agents[1]
            self.judge_agent = agents[2]
            if self.verbose:
                logger.info("Using agents from provided list")
            return

        # Option 2: Use individual agent parameters
        if (
            pro_agent is not None
            and con_agent is not None
            and judge_agent is not None
        ):
            self.pro_agent = pro_agent
            self.con_agent = con_agent
            self.judge_agent = judge_agent
            if self.verbose:
                logger.info("Using individually provided agents")
            return

        # Option 3: Create preset agents
        if preset_agents:
            self._create_preset_agents()
            if self.verbose:
                logger.info("Using preset agents")
            return

        # No valid configuration
        raise ValueError(
            "No valid agent configuration provided. Either:\n"
            "1. Provide all three agents: pro_agent, con_agent, judge_agent\n"
            "2. Provide an agents list with exactly 3 agents: agents=[pro, con, judge]\n"
            "3. Set preset_agents=True to use default agents"
        )

    def _create_preset_agents(self) -> None:
        """
        Create preset agents with default configurations.

        Creates three agents (Pro, Con, Judge) with predefined system prompts
        optimized for debate scenarios.
        """
        self.pro_agent = Agent(
            agent_name="Pro-Debater",
            agent_description="Expert debater arguing in favor of propositions",
            system_prompt=PRO_AGENT_SYSTEM_PROMPT,
            model_name=self.model_name,
            max_loops=1,
            verbose=self.verbose,
        )

        self.con_agent = Agent(
            agent_name="Con-Debater",
            agent_description="Expert debater arguing against propositions",
            system_prompt=CON_AGENT_SYSTEM_PROMPT,
            model_name=self.model_name,
            max_loops=1,
            verbose=self.verbose,
        )

        self.judge_agent = Agent(
            agent_name="Debate-Judge",
            agent_description="Impartial judge evaluating debate arguments",
            system_prompt=JUDGE_AGENT_SYSTEM_PROMPT,
            model_name=self.model_name,
            max_loops=1,
            verbose=self.verbose,
        )

    def run(self, task: str) -> Union[str, List, dict]:
        """
        Execute the debate with judge refinement process.

        Args:
            task (str): The initial topic or question to debate.

        Returns:
            Union[str, List, dict]: The formatted conversation history or final refined answer,
                depending on output_type.

        Raises:
            ValueError: If task is None or empty.
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        # Initialize agents with their roles
        self._initialize_agents(task)

        # Start with the initial task
        current_topic = task

        if self.verbose:
            logger.info(f"Starting debate on: {task}")

        # Execute N loops of debate and refinement
        for round_num in range(self.max_loops):
            if self.verbose:
                logger.info(f"Loop {round_num + 1}/{self.max_loops}")

            # Step 1: Pro agent presents argument
            pro_prompt = self._create_pro_prompt(
                current_topic, round_num
            )
            pro_argument = self.pro_agent.run(task=pro_prompt)
            self.conversation.add(
                self.pro_agent.agent_name, pro_argument
            )

            if self.verbose:
                logger.debug(f"Pro argument: {pro_argument[:100]}...")

            # Step 2: Con agent presents counter-argument
            con_prompt = self._create_con_prompt(
                current_topic, pro_argument, round_num
            )
            con_argument = self.con_agent.run(task=con_prompt)
            self.conversation.add(
                self.con_agent.agent_name, con_argument
            )

            if self.verbose:
                logger.debug(f"Con argument: {con_argument[:100]}...")

            # Step 3: Judge evaluates both arguments and provides synthesis
            judge_prompt = self._create_judge_prompt(
                current_topic, pro_argument, con_argument, round_num
            )
            judge_synthesis = self.judge_agent.run(task=judge_prompt)
            self.conversation.add(
                self.judge_agent.agent_name, judge_synthesis
            )

            if self.verbose:
                logger.debug(
                    f"Judge synthesis: {judge_synthesis[:100]}..."
                )

            # Use judge's synthesis as input for next loop
            current_topic = judge_synthesis

        # Return formatted output
        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def _initialize_agents(self, task: str) -> None:
        """
        Initialize agents with their respective roles and context.

        Args:
            task (str): The initial task/topic for context.
        """
        # Initialize Pro agent
        pro_intro = (
            f"You are {self.pro_agent.agent_name}, arguing in favor (Pro position) "
            f"of the topic: {task}. Your role is to present strong, well-reasoned "
            f"arguments supporting your position. You will debate against "
            f"{self.con_agent.agent_name}, who will argue against your position. "
            f"A judge ({self.judge_agent.agent_name}) will evaluate both arguments "
            f"and provide synthesis. Present compelling evidence and reasoning."
        )
        self.pro_agent.run(task=pro_intro)

        # Initialize Con agent
        con_intro = (
            f"You are {self.con_agent.agent_name}, arguing against (Con position) "
            f"of the topic: {task}. Your role is to present strong, well-reasoned "
            f"counter-arguments. You will debate against {self.pro_agent.agent_name}, "
            f"who will argue in favor. A judge ({self.judge_agent.agent_name}) will "
            f"evaluate both arguments and provide synthesis. Present compelling "
            f"counter-evidence and reasoning."
        )
        self.con_agent.run(task=con_intro)

        # Initialize Judge agent
        judge_intro = (
            f"You are {self.judge_agent.agent_name}, an impartial judge evaluating "
            f"a debate between {self.pro_agent.agent_name} (Pro) and "
            f"{self.con_agent.agent_name} (Con) on the topic: {task}. "
            f"Your role is to carefully evaluate both arguments, identify strengths "
            f"and weaknesses, and provide a refined synthesis that incorporates the "
            f"best elements from both sides. You may declare a winner or provide a "
            f"balanced synthesis. Your output will be used to refine the discussion "
            f"in subsequent loops."
        )
        self.judge_agent.run(task=judge_intro)

    def _create_pro_prompt(self, topic: str, round_num: int) -> str:
        """
        Create the prompt for the Pro agent.

        Args:
            topic (str): The current topic or refined question.
            round_num (int): The current loop number (0-indexed).

        Returns:
            str: The prompt for the Pro agent.
        """
        if round_num == 0:
            return (
                f"Present your argument in favor of: {topic}\n\n"
                f"Provide a strong, well-reasoned argument with evidence and examples."
            )
        else:
            return (
                f"Loop {round_num + 1}: Based on the judge's previous evaluation, "
                f"present an improved argument in favor of: {topic}\n\n"
                f"Address any weaknesses identified and strengthen your position "
                f"with additional evidence and reasoning."
            )

    def _create_con_prompt(
        self, topic: str, pro_argument: str, round_num: int
    ) -> str:
        """
        Create the prompt for the Con agent.

        Args:
            topic (str): The current topic or refined question.
            pro_argument (str): The Pro agent's argument to counter.
            round_num (int): The current loop number (0-indexed).

        Returns:
            str: The prompt for the Con agent.
        """
        if round_num == 0:
            return (
                f"Present your counter-argument against: {topic}\n\n"
                f"Pro's argument:\n{pro_argument}\n\n"
                f"Provide a strong, well-reasoned counter-argument that addresses "
                f"the Pro's points and presents evidence against the position."
            )
        else:
            return (
                f"Loop {round_num + 1}: Based on the judge's previous evaluation, "
                f"present an improved counter-argument against: {topic}\n\n"
                f"Pro's current argument:\n{pro_argument}\n\n"
                f"Address any weaknesses identified and strengthen your counter-position "
                f"with additional evidence and reasoning."
            )

    def _create_judge_prompt(
        self,
        topic: str,
        pro_argument: str,
        con_argument: str,
        round_num: int,
    ) -> str:
        """
        Create the prompt for the Judge agent.

        Args:
            topic (str): The current topic or refined question.
            pro_argument (str): The Pro agent's argument.
            con_argument (str): The Con agent's argument.
            round_num (int): The current loop number (0-indexed).

        Returns:
            str: The prompt for the Judge agent.
        """
        is_final_round = round_num == self.max_loops - 1

        prompt = (
            f"Loop {round_num + 1}/{self.max_loops}: Evaluate the debate on: {topic}\n\n"
            f"Pro's argument ({self.pro_agent.agent_name}):\n{pro_argument}\n\n"
            f"Con's argument ({self.con_agent.agent_name}):\n{con_argument}\n\n"
        )

        if is_final_round:
            prompt += (
                "This is the final loop. Provide a comprehensive final evaluation:\n"
                "- Identify the strongest points from both sides\n"
                "- Determine a winner OR provide a balanced synthesis\n"
                "- Present a refined, well-reasoned answer that incorporates the best "
                "elements from both arguments\n"
                "- This will be the final output of the debate"
            )
        else:
            prompt += (
                "Evaluate both arguments and provide:\n"
                "- Assessment of strengths and weaknesses in each argument\n"
                "- A refined synthesis that incorporates the best elements from both sides\n"
                "- Specific feedback for improvement in the next loop\n"
                "- Your synthesis will be used as the topic for the next loop"
            )

        return prompt

    def get_conversation_history(self) -> List[dict]:
        """
        Get the full conversation history.

        Returns:
            List[dict]: List of message dictionaries containing the conversation history.
        """
        return self.conversation.return_messages_as_list()

    def get_final_answer(self) -> str:
        """
        Get the final refined answer from the judge.

        Returns:
            str: The content of the final judge synthesis.
        """
        return self.conversation.get_final_message_content()

    def batched_run(self, tasks: List[str]) -> List[str]:
        """
        Run the debate with judge refinement process for a batch of tasks.

        Args:
            tasks (List[str]): The list of tasks to run the debate with judge refinement process for.

        Returns:
            List[str]: The list of final refined answers.
        """
        return [self.run(task) for task in tasks]
