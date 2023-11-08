import random
import tenacity
from langchain.output_parsers import RegexParser


# utils
class BidOutputParser(RegexParser):

    def get_format_instructions(self) -> str:
        return (
            "Your response should be an integrater delimited by angled brackets like"
            " this: <int>")


bid_parser = BidOutputParser(regex=r"<(\d+)>",
                             output_keys=["bid"],
                             default_output_key="bid")


def select_next_speaker(step: int, agents, director) -> int:
    # if the step if even => director
    # => director selects next speaker
    if step % 2 == 1:
        idx = 0
    else:
        idx = director.select_next_speaker() + 1
    return idx


# main
class MultiAgentCollaboration:

    def __init__(
        self,
        agents,
        selection_function,
    ):
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.run(f"Name {name} and message: {message}")
        self._step += 1

    def step(self) -> tuple[str, str]:
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        message = speaker.send()
        message = speaker.send()

        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        self._step += 1
        return speaker.name, message

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
        bid_string = agent.bid()
        bid = int(bid_parser.parse(bid_string)["bid"])
        return bid

    def select_next_speaker(
        self,
        step: int,
        agents,
    ) -> int:
        bids = []
        for agent in agents:
            bid = self.ask_for_bid(agent)
            bids.append(bid)
        max_value = max(bids)
        max_indices = [i for i, x in enumerate(bids) if x == max_value]
        idx = random.choice(max_indices)
        return idx

    def run(self, max_iters: int = 10):
        n = 0
        self.reset()
        self.inject("Debate Moderator")
        print("(Debate Moderator): ")
        print("\n")

        while n < max_iters:
            name, message = self.step()
            print(f"({name}): {message}")
            print("\n")
            n += 1
