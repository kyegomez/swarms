"""Every vote and the consensus, recorded in order."""

from swarms import Agent, FileChannel, MajorityVoting

voters = [
    Agent(agent_name=f"Voter-{i}", model_name="gpt-5.4", max_loops=1)
    for i in range(3)
]

channel = FileChannel("runs/majority.md")

vote = MajorityVoting(agents=voters, max_loops=1, channel=channel)
vote.run(
    "Is Rust or Go the better choice for a high-throughput API gateway? Answer with one word, then one sentence."
)

print(channel.path.read_text())
