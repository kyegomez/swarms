"""Test auction selection without constructing model-backed agents."""

import ast
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/multi_agent/social_algorithms_examples"
    / "auction_market_algorithm_example.py"
)


def load_auction_algorithm():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]
    namespace = {}
    exec(  # noqa: S102 - only local example functions
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["auction_market_algorithm"]


class RecordingAgent:
    def __init__(self, name, response):
        self.agent_name = name
        self.response = response
        self.prompts = []

    def run(self, prompt):
        self.prompts.append(prompt)
        return self.response


class AuctionWinnerTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)
        self.algorithm = load_auction_algorithm()

    def auction(self, response, names=("Technical", "Business")):
        bidders = [
            RecordingAgent(name, f"proposal-{name}") for name in names
        ]
        auctioneer = RecordingAgent("Auctioneer", response)
        assessor = RecordingAgent("Assessor", "quality assessment")
        result = self.algorithm(
            [*bidders, auctioneer, assessor], "Offline task"
        )
        return result, bidders, auctioneer

    def test_winner_line_beats_earlier_mentions_and_ranking(self):
        result, bidders, _ = self.auction(
            "Technical has a strong proposal.\n"
            "Winner: Business\nRanking: Business, Technical"
        )
        self.assertEqual(result["winner"], "bidder_2_Business")
        self.assertEqual(len(bidders[0].prompts), 1)
        self.assertEqual(len(bidders[1].prompts), 2)
        self.assertIn(
            "Your Winning Proposal: proposal-Business",
            bidders[1].prompts[-1],
        )

    def test_third_bidder_receives_its_contract(self):
        result, bidders, _ = self.auction(
            "Winner: Creative\n"
            "Ranking: Technical, Business, Creative",
            names=("Technical", "Business", "Creative"),
        )
        self.assertEqual(result["winner"], "bidder_3_Creative")
        self.assertEqual(
            [len(bidder.prompts) for bidder in bidders], [1, 1, 2]
        )

    def test_agent_names_are_matched_exactly(self):
        result, bidders, _ = self.auction(
            "Winner: Anna", names=("Ann", "Anna")
        )
        self.assertEqual(result["winner"], "bidder_2_Anna")
        self.assertEqual(len(bidders[0].prompts), 1)

    def test_unknown_winner_does_not_award_a_mentioned_bidder(self):
        result, bidders, _ = self.auction(
            "Winner: Unknown\nTechnical and Business were considered."
        )
        self.assertIsNone(result["winner"])
        self.assertEqual(
            result["contract"], "No clear winner identified"
        )
        self.assertEqual(
            [len(bidder.prompts) for bidder in bidders], [1, 1]
        )

    def test_missing_selection_does_not_award_the_first_mention(self):
        result, bidders, _ = self.auction(
            "Ranking: Technical, Business. No winner was selected."
        )
        self.assertIsNone(result["winner"])
        self.assertEqual(
            [len(bidder.prompts) for bidder in bidders], [1, 1]
        )

    def test_plain_winner_line_and_auction_feedback_are_preserved(
        self,
    ):
        response = "Winner: Technical\nSelected for feasibility."
        result, bidders, _ = self.auction(response)
        self.assertEqual(result["winner"], "bidder_1_Technical")
        self.assertEqual(result["auction_result"], response)
        self.assertIn(response, bidders[0].prompts[-1])


if __name__ == "__main__":
    unittest.main()
