"""
Debate/Self-Refinement with Judge Architecture

This module implements a debate architecture where two agents (Pro and Con)
debate a topic, and a Judge agent evaluates their arguments and provides
refined synthesis. The process repeats for N rounds to progressively refine
the answer.
"""

from typing import List, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class DebateWithJudge:
    """
    A debate architecture with self-refinement through a judge agent.

    This class implements a debate system where:
    1. Agent A (Pro) and Agent B (Con) present opposing arguments
    2. Both arguments are evaluated by a Judge/Critic Agent
    3. The Judge provides a winner or synthesis → refined answer
    4. The process repeats for N rounds to progressively improve the answer

    Architecture:
        Agent A (Pro) ↔ Agent B (Con)
              │            │
              ▼            ▼
           Judge / Critic Agent
              │
              ▼
        Winner or synthesis → refined answer

    Attributes:
        pro_agent (Agent): The agent arguing in favor (Pro position).
        con_agent (Agent): The agent arguing against (Con position).
        judge_agent (Agent): The judge agent that evaluates arguments and provides synthesis.
        max_rounds (int): Maximum number of debate rounds to execute.
        output_type (str): Format for the output conversation history.
        verbose (bool): Whether to enable verbose logging.
    """

    def __init__(
        self,
        pro_agent: Agent,
        con_agent: Agent,
        judge_agent: Agent,
        max_rounds: int = 3,
        output_type: str = "str-all-except-first",
        verbose: bool = True,
    ):
        """
        Initialize the DebateWithJudge architecture.

        Args:
            pro_agent (Agent): The agent arguing in favor (Pro position).
            con_agent (Agent): The agent arguing against (Con position).
            judge_agent (Agent): The judge agent that evaluates arguments and provides synthesis.
            max_rounds (int): Maximum number of debate rounds to execute. Defaults to 3.
            output_type (str): Format for the output conversation history. Defaults to "str-all-except-first".
            verbose (bool): Whether to enable verbose logging. Defaults to True.

        Raises:
            ValueError: If any of the required agents are None or if max_rounds is less than 1.
        """
        if pro_agent is None:
            raise ValueError("pro_agent cannot be None")
        if con_agent is None:
            raise ValueError("con_agent cannot be None")
        if judge_agent is None:
            raise ValueError("judge_agent cannot be None")
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        self.pro_agent = pro_agent
        self.con_agent = con_agent
        self.judge_agent = judge_agent
        self.max_rounds = max_rounds
        self.output_type = output_type
        self.verbose = verbose

        # Initialize conversation history
        self.conversation = Conversation()

        if self.verbose:
            logger.info(
                f"DebateWithJudge initialized with {max_rounds} rounds"
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

        # Execute N rounds of debate and refinement
        for round_num in range(self.max_rounds):
            if self.verbose:
                logger.info(
                    f"Round {round_num + 1}/{self.max_rounds}"
                )

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

            # Use judge's synthesis as input for next round
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
            f"in subsequent rounds."
        )
        self.judge_agent.run(task=judge_intro)

    def _create_pro_prompt(self, topic: str, round_num: int) -> str:
        """
        Create the prompt for the Pro agent.

        Args:
            topic (str): The current topic or refined question.
            round_num (int): The current round number (0-indexed).

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
                f"Round {round_num + 1}: Based on the judge's previous evaluation, "
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
            round_num (int): The current round number (0-indexed).

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
                f"Round {round_num + 1}: Based on the judge's previous evaluation, "
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
            round_num (int): The current round number (0-indexed).

        Returns:
            str: The prompt for the Judge agent.
        """
        is_final_round = round_num == self.max_rounds - 1

        prompt = (
            f"Round {round_num + 1}/{self.max_rounds}: Evaluate the debate on: {topic}\n\n"
            f"Pro's argument ({self.pro_agent.agent_name}):\n{pro_argument}\n\n"
            f"Con's argument ({self.con_agent.agent_name}):\n{con_argument}\n\n"
        )

        if is_final_round:
            prompt += (
                "This is the final round. Provide a comprehensive final evaluation:\n"
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
                "- Specific feedback for improvement in the next round\n"
                "- Your synthesis will be used as the topic for the next round"
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
