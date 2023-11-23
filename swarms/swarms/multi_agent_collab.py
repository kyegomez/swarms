import json
import random
from typing import List

import tenacity
from langchain.output_parsers import RegexParser

from swarms.structs.flow import Flow
from swarms.utils.logger import logger


# utils
class BidOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return (
            "Your response should be an integrater delimited by angled brackets like"
            " this: <int>"
        )


bid_parser = BidOutputParser(
    regex=r"<(\d+)>", output_keys=["bid"], default_output_key="bid"
)


def select_next_speaker_director(step: int, agents, director) -> int:
    # if the step if even => director
    # => director selects next speaker
    if step % 2 == 1:
        idx = 0
    else:
        idx = director.select_next_speaker() + 1
    return idx


# Define a selection function
def select_speaker_round_table(step: int, agents) -> int:
    # This function selects the speaker in a round-robin fashion
    return step % len(agents)


# main
class MultiAgentCollaboration:
    """
    Multi-agent collaboration class.

    Attributes:
        agents (List[Flow]): The agents in the collaboration.
        selection_function (callable): The function that selects the next speaker.
            Defaults to select_next_speaker.
        max_iters (int): The maximum number of iterations. Defaults to 10.

    Methods:
        reset: Resets the state of all agents.
        inject: Injects a message into the collaboration.
        inject_agent: Injects an agent into the collaboration.
        step: Steps through the collaboration.
        ask_for_bid: Asks an agent for a bid.
        select_next_speaker: Selects the next speaker.
        run: Runs the collaboration.
        format_results: Formats the results of the run method.


    Usage:
    >>> from swarms.models import MultiAgentCollaboration
    >>> from swarms.models import Flow
    >>> from swarms.models import OpenAIChat
    >>> from swarms.models import Anthropic


    """

    def __init__(
        self,
        agents: List[Flow],
        selection_function: callable = select_next_speaker_director,
        max_iters: int = 10,
        autosave: bool = True,
        saved_file_path_name: str = "multi_agent_collab.json",
        stopping_token: str = "<DONE>",
        logging: bool = True,
    ):
        self.agents = agents
        self.select_next_speaker = selection_function
        self._step = 0
        self.max_iters = max_iters
        self.autosave = autosave
        self.saved_file_path_name = saved_file_path_name
        self.stopping_token = stopping_token
        self.results = []
        self.logger = logger
        self.logging = logging

    def reset(self):
        """Resets the state of all agents."""
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """Injects a message into the multi-agent collaboration."""
        for agent in self.agents:
            agent.run(f"Name {name} and message: {message}")
        self._step += 1

    def inject_agent(self, agent: Flow):
        """Injects an agent into the multi-agent collaboration."""
        self.agents.append(agent)

    def step(self) -> tuple[str, str]:
        """Steps through the multi-agent collaboration."""
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        message = speaker.send()
        message = speaker.send()

        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        self._step += 1

        if self.logging:
            self.log_step(speaker, message)

        return speaker.name, message

    def log_step(self, speaker: str, response: str):
        """Logs the step of the multi-agent collaboration."""
        self.logger.info(f"{speaker.name}: {response}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occured: {retry_state.outcome.exception()}, retying..."
        ),
        retry_error_callback=lambda retry_state: 0,
    )
    def ask_for_bid(self, agent) -> str:
        """Asks an agent for a bid."""
        bid_string = agent.bid()
        bid = int(bid_parser.parse(bid_string)["bid"])
        return bid

    def select_next_speaker(
        self,
        step: int,
        agents,
    ) -> int:
        """Selects the next speaker."""
        bids = []
        for agent in agents:
            bid = self.ask_for_bid(agent)
            bids.append(bid)
        max_value = max(bids)
        max_indices = [i for i, x in enumerate(bids) if x == max_value]
        idx = random.choice(max_indices)
        return idx

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occured: {retry_state.outcome.exception()}, retying..."
        ),
        retry_error_callback=lambda retry_state: 0,
    )
    def run(self):
        """Runs the multi-agent collaboration."""
        n = 0
        self.reset()
        self.inject("Debate Moderator")
        print("(Debate Moderator): ")
        print("\n")

        while n < self.max_iters:
            name, message = self.step()
            print(f"({name}): {message}")
            print("\n")
            n += 1

    def format_results(self, results):
        """Formats the results of the run method"""
        formatted_results = "\n".join(
            [f"{result['agent']} responded: {result['response']}" for result in results]
        )
        return formatted_results

    def save(self):
        """Saves the state of all agents."""
        state = {
            "step": self._step,
            "results": [
                {"agent": r["agent"].name, "response": r["response"]}
                for r in self.results
            ],
        }

        with open(self.saved_file_path_name, "w") as file:
            json.dump(state, file)

    def load(self):
        """Loads the state of all agents."""
        with open(self.saved_file_path_name, "r") as file:
            state = json.load(file)
        self._step = state["step"]
        self.results = state["results"]
        return state

    def __repr__(self):
        return f"MultiAgentCollaboration(agents={self.agents}, selection_function={self.select_next_speaker}, max_iters={self.max_iters}, autosave={self.autosave}, saved_file_path_name={self.saved_file_path_name})"

    def performance(self):
        """Tracks and reports the performance of each agent"""
        performance_data = {}
        for agent in self.agents:
            performance_data[agent.name] = agent.get_performance_metrics()
        return performance_data

    def set_interaction_rules(self, rules):
        """Sets the interaction rules for each agent"""
        self.interaction_rules = rules
