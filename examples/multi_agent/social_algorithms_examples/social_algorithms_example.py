from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms

load_dotenv()

researcher = Agent(
    agent_name="Researcher",
    agent_description="Gathers evidence and lays out the factual landscape",
    system_prompt=(
        "You are a research analyst. Given a question, lay out the key facts, "
        "the main competing positions, and the evidence behind each. Be concrete "
        "and cite the reasoning behind every claim. Keep it under 300 words."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

critic = Agent(
    agent_name="Critic",
    agent_description="Attacks the research for gaps and weak reasoning",
    system_prompt=(
        "You are a skeptical reviewer. Given a piece of research, find its "
        "weakest claims, its unstated assumptions, and anything important it "
        "left out. Be specific and brief. Do not rewrite it yourself."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Produces the final answer from the research and critique",
    system_prompt=(
        "You are an editor. Given research and a critique of it, write the "
        "final answer that survives the critique. Address every objection "
        "raised, or say plainly why it does not change the conclusion."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)


def peer_review(agents, task, rounds: int = 1, **kwargs):
    """Research a question, critique the research, then edit past the critique.

    This is the "social algorithm" — plain Python that decides who speaks, in
    what order, and what each one sees. Anything callable with this signature
    works, so the communication pattern is yours to define.

    Args:
        agents: The agents given to SocialAlgorithms, in declaration order.
        task: The question to answer.
        rounds: How many research/critique cycles to run before editing.
        **kwargs: Ignored here; forwarded by SocialAlgorithms.

    Returns:
        The editor's final answer.
    """
    researcher, critic, editor = agents

    draft = researcher.run(f"Research this question: {task}")

    for _ in range(rounds):
        critique = critic.run(
            f"Critique this research on '{task}':\n\n{draft}"
        )
        draft = researcher.run(
            f"Revise your research to answer this critique.\n\n"
            f"Research:\n{draft}\n\nCritique:\n{critique}"
        )

    return editor.run(
        f"Write the final answer to '{task}' from this research:\n\n{draft}"
    )


social_alg = SocialAlgorithms(
    name="Peer-Review",
    description="Research, critique, revise, then edit into a final answer",
    agents=[researcher, critic, editor],
    social_algorithm=peer_review,
    max_execution_time=600,
    verbose=True,
)

result = social_alg.run(
    task="Are vector databases actually necessary for production RAG, or is Postgres with pgvector enough?",
    algorithm_args={"rounds": 2},
)

print("\n=== FINAL ANSWER ===")
print(result.final_outputs["result"])

print("\n=== RUN STATS ===")
print(f"agent messages: {result.total_steps}")
print(f"elapsed:        {result.execution_time:.1f}s")

# Every agent call is recorded automatically — the algorithm reports nothing.
print("\n=== TRANSCRIPT ===")
for message in social_alg.conversation.conversation_history:
    print(f"\n[{message['role']}]\n{message['content']}")
