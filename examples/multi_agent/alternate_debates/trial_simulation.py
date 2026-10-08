"""
TrialSimulation — Simulate a legal trial with structured phases and roles.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/trial_simulation.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class TrialSimulation:
    """
    Simulate a legal trial with structured phases and roles.
    """

    def __init__(
        self,
        prosecution: Agent = None,
        defense: Agent = None,
        judge: Agent = None,
        witnesses: List[Agent] = None,
        phases: List[str] = None,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the trial simulation structure.

        Args:
            prosecution (Agent): The prosecution attorney agent.
            defense (Agent): The defense attorney agent.
            judge (Agent): The judge agent who presides over the trial.
            witnesses (List[Agent]): List of witness agents.
            phases (List[str]): List of trial phases to simulate.
            output_type (str): Output format for conversation history.
        """
        self.prosecution = prosecution
        self.defense = defense
        self.judge = judge
        self.witnesses = witnesses
        self.phases = phases
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the trial simulation.

        Args:
            task (str): Description of the legal case.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.prosecution or not self.defense or not self.judge:
            raise ValueError(
                "Prosecution, defense, and judge agents are all required."
            )

        if not self.phases:
            self.phases = ["opening", "testimony", "cross", "closing"]

        # Create trial participant list for context
        witness_names = [
            witness.agent_name for witness in (self.witnesses or [])
        ]
        trial_participants = f"Prosecution: {self.prosecution.agent_name}. Defense: {self.defense.agent_name}. Judge: {self.judge.agent_name}."
        if witness_names:
            trial_participants += (
                f" Witnesses: {', '.join(witness_names)}."
            )

        # Inform judge about all participants
        judge_intro = f"You are {self.judge.agent_name}, presiding over this trial. {trial_participants} Maintain order and ensure proper legal procedure."
        self.judge.run(task=judge_intro)

        # Inform prosecution about trial setup
        prosecution_intro = f"You are {self.prosecution.agent_name}, prosecuting attorney. {trial_participants} Present the case for the prosecution professionally."
        self.prosecution.run(task=prosecution_intro)

        # Inform defense about trial setup
        defense_intro = f"You are {self.defense.agent_name}, defense attorney. {trial_participants} Defend your client professionally."
        self.defense.run(task=defense_intro)

        # Inform witnesses about their role
        for witness in self.witnesses or []:
            witness_intro = f"You are {witness.agent_name}, a witness in this trial. {trial_participants} Provide truthful testimony when called."
            witness.run(task=witness_intro)

        current_case = task

        for phase in self.phases:
            # Judge opens the phase
            phase_opening = (
                f"Phase: {phase.upper()}. Case: {current_case}"
            )
            judge_opening = self.judge.run(task=phase_opening)
            conversation.add(self.judge.agent_name, judge_opening)

            if phase == "opening":
                # Prosecution opening statement
                prosecution_opening = self.prosecution.run(
                    task=f"Give opening statement for: {current_case}"
                )
                conversation.add(
                    self.prosecution.agent_name, prosecution_opening
                )

                # Defense opening statement
                defense_opening = self.defense.run(
                    task=f"Give opening statement responding to: {prosecution_opening}"
                )
                conversation.add(
                    self.defense.agent_name, defense_opening
                )

            elif phase == "testimony" and self.witnesses:
                # Witness testimony
                for i, witness in enumerate(self.witnesses):
                    witness_testimony = witness.run(
                        task=f"Provide testimony for: {current_case}"
                    )
                    conversation.add(
                        witness.agent_name, witness_testimony
                    )

            elif phase == "cross":
                # Cross-examination
                for witness in self.witnesses or []:
                    cross_exam = self.prosecution.run(
                        task=f"Cross-examine this testimony: {witness_testimony}"
                    )
                    conversation.add(
                        self.prosecution.agent_name, cross_exam
                    )

                    redirect = self.defense.run(
                        task=f"Redirect examination: {cross_exam}"
                    )
                    conversation.add(
                        self.defense.agent_name, redirect
                    )

            elif phase == "closing":
                # Closing arguments
                prosecution_closing = self.prosecution.run(
                    task="Give closing argument"
                )
                conversation.add(
                    self.prosecution.agent_name, prosecution_closing
                )

                defense_closing = self.defense.run(
                    task=f"Give closing argument responding to: {prosecution_closing}"
                )
                conversation.add(
                    self.defense.agent_name, defense_closing
                )

                # Judge's verdict
                verdict_prompt = f"Render verdict based on: {[msg['content'] for msg in conversation.conversation_history[-2:]]}"
                verdict = self.judge.run(task=verdict_prompt)
                conversation.add(self.judge.agent_name, verdict)

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    prosecution = Agent(
        agent_name="Prosecution",
        agent_description="Prosecuting attorney",
        system_prompt="You are the prosecuting attorney. Build the case methodically from the evidence.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    defense = Agent(
        agent_name="Defense",
        agent_description="Defense attorney",
        system_prompt="You are the defense attorney. Challenge the evidence and protect your client's interests.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    judge = Agent(
        agent_name="Judge",
        agent_description="Presiding judge",
        system_prompt="You are the presiding judge. Maintain procedure and render a reasoned verdict.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    witnesses = [
        Agent(
            agent_name="Expert-Witness",
            agent_description="Technical expert witness",
            system_prompt="You are a technical expert witness. Testify truthfully within your expertise.",
            model_name="gpt-5.4",
            max_loops=1,
        )
    ]

    # NOTE: keep "testimony" in `phases` before "cross" -- the cross phase reads
    # the testimony produced by the testimony phase.
    trial = TrialSimulation(
        prosecution=prosecution,
        defense=defense,
        judge=judge,
        witnesses=witnesses,
        phases=["opening", "testimony", "cross", "closing"],
    )

    print(
        trial.run(
            "Contract dispute: a vendor delivered a system that failed load testing at "
            "60% of the contractually specified throughput."
        )
    )
