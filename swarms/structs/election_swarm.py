import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.conversation import Conversation


def _create_voting_prompt(candidate_agents: List[Agent]) -> str:
    """
    Create a comprehensive voting prompt for the election.

    This method generates a detailed prompt that instructs voter agents on:
    - Available candidates
    - Required structured output format
    - Evaluation criteria
    - Voting guidelines

    Returns:
        str: A formatted voting prompt string
    """
    candidate_names = [
        (agent.agent_name if hasattr(agent, "agent_name") else str(i))
        for i, agent in enumerate(candidate_agents)
    ]

    prompt = f"""
    You are participating in an election to choose the best candidate agent.
    
    Available candidates: {', '.join(candidate_names)}
    
    Please vote for one candidate and provide your reasoning with the following structured output:
    
    1. rationality: A detailed explanation of the reasoning behind your decision. Include logical considerations, supporting evidence, and trade-offs that were evaluated when selecting this candidate.
    
    2. self_interest: A comprehensive discussion of how self-interest influenced your decision, if at all. Explain whether personal or role-specific incentives played a role, or if your choice was primarily for the collective benefit of the swarm.
    
    3. candidate_agent_name: The full name or identifier of the candidate you are voting for. This should exactly match one of the available candidate names listed above.
    
    Consider the candidates' capabilities, experience, and alignment with the swarm's objectives when making your decision.
    """

    print(prompt)

    return prompt


def get_vote_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "vote",
                "description": "Cast a vote for a CEO candidate with reasoning and self-interest analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rationality": {
                            "type": "string",
                            "description": "A detailed explanation of the reasoning behind this voting decision.",
                        },
                        "self_interest": {
                            "type": "string",
                            "description": "A comprehensive discussion of how self-interest factored into the decision.",
                        },
                        "candidate_agent_name": {
                            "type": "string",
                            "description": "The full name or identifier of the chosen candidate.",
                        },
                    },
                    "required": [
                        "rationality",
                        "self_interest",
                        "candidate_agent_name",
                    ],
                },
            },
        }
    ]


class ElectionSwarm:
    """
    A swarm system that conducts elections among multiple agents to choose the best candidate.

    The ElectionSwarm orchestrates a voting process where multiple voter agents evaluate
    and vote for candidate agents based on their capabilities, experience, and alignment
    with swarm objectives. The system uses structured output to ensure consistent voting
    format and provides detailed reasoning for each vote.

    Attributes:
        id (str): Unique identifier for the election swarm
        name (str): Name of the election swarm
        description (str): Description of the election swarm's purpose
        max_loops (int): Maximum number of voting rounds (default: 1)
        agents (List[Agent]): List of voter agents that will participate in the election
        candidate_agents (List[Agent]): List of candidate agents to be voted on
        kwargs (dict): Additional keyword arguments
        show_dashboard (bool): Whether to display the election dashboard
        conversation (Conversation): Conversation history for the election
    """

    def __init__(
        self,
        name: str = "Election Swarm",
        description: str = "An election swarm is a swarm of agents that will vote on a candidate.",
        agents: Union[List[Agent], List[Callable]] = None,
        candidate_agents: Union[List[Agent], List[Callable]] = None,
        id: str = str(uuid.uuid4()),
        max_loops: int = 1,
        show_dashboard: bool = True,
        **kwargs,
    ):
        """
        Initialize the ElectionSwarm.

        Args:
            name (str, optional): Name of the election swarm
            description (str, optional): Description of the election swarm's purpose
            agents (Union[List[Agent], List[Callable]], optional): List of voter agents
            candidate_agents (Union[List[Agent], List[Callable]], optional): List of candidate agents
            id (str, optional): Unique identifier for the election swarm
            max_loops (int, optional): Maximum number of voting rounds (default: 1)
            show_dashboard (bool, optional): Whether to display the election dashboard (default: True)
            **kwargs: Additional keyword arguments
        """
        self.id = id
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.agents = agents
        self.candidate_agents = candidate_agents
        self.kwargs = kwargs
        self.show_dashboard = show_dashboard
        self.conversation = Conversation()

        self.reliability_check()

        self.setup_voter_agents()

    def reliability_check(self):
        """
        Check the reliability of the voter agents.
        """
        if self.agents is None:
            raise ValueError("Voter agents are not set")

        if self.candidate_agents is None:
            raise ValueError("Candidate agents are not set")

        if self.max_loops is None or self.max_loops < 1:
            raise ValueError("Max loops are not set")

    def setup_concurrent_workflow(self):
        """
        Create a concurrent workflow for running voter agents in parallel.

        Returns:
            ConcurrentWorkflow: A configured concurrent workflow for the election
        """
        return ConcurrentWorkflow(
            name=self.name,
            description=self.description,
            agents=self.agents,
            output_type="dict-all-except-first",
            show_dashboard=self.show_dashboard,
        )

    def run_voter_agents(
        self, task: str, img: Optional[str] = None, *args, **kwargs
    ):
        """
        Execute the voting process by running all voter agents concurrently.

        Args:
            task (str): The election task or question to be voted on
            img (Optional[str], optional): Image path if visual voting is required
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List[Dict[str, Any]]: Results from all voter agents containing their votes and reasoning
        """
        concurrent_workflow = self.setup_concurrent_workflow()

        results = concurrent_workflow.run(
            task=task, img=img, *args, **kwargs
        )

        conversation_history = (
            concurrent_workflow.conversation.conversation_history
        )

        for message in conversation_history:
            self.conversation.add(
                role=message["role"], content=message["content"]
            )

        return results

    def parse_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Parse voting results to count votes for each candidate.

        Args:
            results (List[Dict[str, Any]]): List of voting results from voter agents

        Returns:
            Dict[str, int]: Dictionary mapping candidate names to their vote counts
        """
        # Count the number of votes for each candidate
        vote_counts = {}
        for result in results:
            candidate_name = result["candidate_agent_name"]
            vote_counts[candidate_name] = (
                vote_counts.get(candidate_name, 0) + 1
            )

        # Find the candidate with the most votes

        return vote_counts

    def run(
        self, task: str, img: Optional[str] = None, *args, **kwargs
    ):
        """
        Execute the complete election process.

        This method orchestrates the entire election by:
        1. Adding the task to the conversation history
        2. Running all voter agents concurrently
        3. Collecting and processing the voting results

        Args:
            task (str): The election task or question to be voted on
            img (Optional[str], optional): Image path if visual voting is required
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List[Dict[str, Any]]: Complete voting results from all agents
        """
        self.conversation.add(role="user", content=task)

        results = self.run_voter_agents(task, img, *args, **kwargs)

        print(results)

        return results

    def setup_voter_agents(self):
        """
        Configure voter agents with structured output capabilities and voting prompts.

        This method sets up each voter agent with:
        - Structured output schema for consistent voting format
        - Voting-specific system prompts
        - Tools for structured response generation

        Returns:
            List[Agent]: Configured voter agents ready for the election
        """
        schema = get_vote_schema()
        prompt = _create_voting_prompt(self.candidate_agents)

        for agent in self.agents:
            agent.tools_list_dictionary = schema
            agent.system_prompt += f"\n\n{prompt}"
