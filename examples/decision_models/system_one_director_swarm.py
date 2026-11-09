import time

from swarms import Agent, DecisionModel, HierarchicalSwarm

TASK = (
    "Should a 20-person startup move its Postgres database from AWS RDS "
    "to a self-managed cluster? Give a recommendation with costs, risks "
    "and a migration plan."
)


class DecisionDirector:
    """
    Director that picks the next worker with a decision model instead of an LLM.

    Args:
        decision_model: Decision model that answers the routing questions.
        workers: Agents the director can assign work to.
        done_threshold: Probability above which the task counts as finished.
        stuck_threshold: Probability above which the last worker counts as stuck.
    """

    agent_name = "System-1-Director"

    def __init__(
        self,
        decision_model: DecisionModel,
        workers: list,
        done_threshold: float = 0.8,
        stuck_threshold: float = 0.7,
    ):
        self.decision_model = decision_model
        self.roster = {
            agent.agent_name: agent.agent_description
            for agent in workers
        }
        self.done_threshold = done_threshold
        self.stuck_threshold = stuck_threshold
        self.transcript = []
        self.last_worker = None
        self.finished = False
        self.latencies = []

    def run(self, task: str, img: str = None) -> dict:
        """
        Decide which worker acts next, or stop when the answer is complete.

        Args:
            task: New messages since the director last ran.
            img: Unused; kept for the director interface.

        Returns:
            Orders for the swarm, empty once the task is done.
        """
        # HierarchicalSwarm only sends what is new, so keep the whole transcript here.
        self.transcript.append(task)
        if self.finished:
            return {"orders": []}

        started = time.perf_counter()
        answers = self.decision_model.run(
            state={
                "transcript": "\n\n".join(self.transcript),
                "last_worker": self.last_worker,
            },
            questions={
                "next_worker": {
                    "type": "choice",
                    "instructions": "Which worker should act next to move the task toward a complete answer?",
                    "criteria": self.roster,
                },
                "done": {
                    "type": "noul",
                    "instructions": "The transcript already contains a complete, final answer to the user's task.",
                },
                "stuck": {
                    "type": "noul",
                    "instructions": "The most recent worker output repeats earlier work without adding anything new.",
                },
            },
        )["answers"]
        latency_ms = (time.perf_counter() - started) * 1000
        self.latencies.append(latency_ms)

        done = answers["done"]["noul"]
        stuck = answers["stuck"]["noul"]
        probabilities = dict(answers["next_worker"]["probabilities"])

        loop = len(self.latencies)
        if done >= self.done_threshold:
            self.finished = True
            print(
                f"loop {loop} | {latency_ms:4.0f} ms | done {done:.2f} -> stop"
            )
            return {"orders": []}

        if stuck >= self.stuck_threshold and self.last_worker:
            probabilities.pop(self.last_worker, None)

        worker = max(probabilities, key=probabilities.get)
        print(
            f"loop {loop} | {latency_ms:4.0f} ms | next {worker} "
            f"({probabilities[worker]:.2f}) | done {done:.2f} | stuck {stuck:.2f}"
        )
        self.last_worker = worker
        return {
            "orders": [
                {
                    "agent_name": worker,
                    "task": (
                        f"As the {worker}, do your part: {self.roster[worker]} "
                        "Build on the team's work so far and keep it concise."
                    ),
                }
            ]
        }


workers = [
    Agent(
        agent_name="Researcher",
        agent_description="Gathers the facts, prices and numbers the answer needs.",
        system_prompt="You research infrastructure questions and report concrete facts and figures.",
        model_name="gpt-5.4-mini",
        max_loops=1,
        print_on=False,
    ),
    Agent(
        agent_name="Analyst",
        agent_description="Weighs trade-offs and turns the facts into a recommendation.",
        system_prompt="You weigh costs, risks and team capacity, then recommend one option.",
        model_name="gpt-5.4-mini",
        max_loops=1,
        print_on=False,
    ),
    Agent(
        agent_name="Writer",
        agent_description="Writes the final answer for the reader: recommendation, costs, risks and plan.",
        system_prompt="You turn the team's notes into a clear, complete final answer.",
        model_name="gpt-5.4-mini",
        max_loops=1,
        print_on=False,
    ),
    Agent(
        agent_name="Critic",
        agent_description="Checks the latest draft for errors, gaps and unsupported claims.",
        system_prompt="You review drafts and list concrete problems, or say the draft is ready.",
        model_name="gpt-5.4-mini",
        max_loops=1,
        print_on=False,
    ),
]

director = DecisionDirector(
    decision_model=DecisionModel(model_name="jev-latest"),
    workers=workers,
)

swarm = HierarchicalSwarm(
    name="System-1-System-2-Swarm",
    director=director,
    agents=workers,
    max_loops=8,
    print_on=False,
)

started = time.perf_counter()
swarm.run(TASK)
total_s = time.perf_counter() - started

worker_names = {agent.agent_name for agent in workers}
worker_turns = [
    message
    for message in swarm.conversation.conversation_history
    if message["role"] in worker_names
]
director_s = sum(director.latencies) / 1000

print(f"\nFinal answer from {worker_turns[-1]['role']}:\n")
print(worker_turns[-1]["content"])
print(
    f"\nSystem 1 (director): {len(director.latencies)} decisions, "
    f"{director_s:.2f} s total, "
    f"{director_s * 1000 / len(director.latencies):.0f} ms each"
)
print(
    f"System 2 (workers): {len(worker_turns)} turns, "
    f"{total_s - director_s:.1f} s total"
)
if director.finished:
    print(
        f"Stopped after {len(worker_turns)} worker turns out of a possible "
        f"{swarm.max_loops}: the director judged the answer complete."
    )
