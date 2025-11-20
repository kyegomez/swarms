"""
Example 1: Policy Debate on AI Regulation

This example demonstrates using DebateWithJudge for a comprehensive policy debate
on AI regulation, with multiple rounds of refinement.
"""

from swarms import Agent, DebateWithJudge

# Create the Pro agent (arguing in favor of AI regulation)
pro_agent = Agent(
    agent_name="Pro-Regulation-Agent",
    system_prompt=(
        "You are a policy expert specializing in technology regulation. "
        "You argue in favor of government regulation of artificial intelligence. "
        "You present well-reasoned arguments focusing on safety, ethics, "
        "and public interest. You use evidence, examples, and logical reasoning. "
        "You are persuasive and articulate, emphasizing the need for oversight "
        "to prevent harm and ensure responsible AI development."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the Con agent (arguing against AI regulation)
con_agent = Agent(
    agent_name="Anti-Regulation-Agent",
    system_prompt=(
        "You are a technology policy expert specializing in innovation. "
        "You argue against heavy government regulation of artificial intelligence. "
        "You present strong counter-arguments focusing on innovation, economic growth, "
        "and the risks of over-regulation. You identify weaknesses in regulatory "
        "proposals and provide compelling alternatives such as industry self-regulation "
        "and ethical guidelines. You emphasize the importance of maintaining "
        "technological competitiveness."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the Judge agent (evaluates and synthesizes)
judge_agent = Agent(
    agent_name="Policy-Judge-Agent",
    system_prompt=(
        "You are an impartial policy analyst and judge who evaluates debates on "
        "technology policy. You carefully analyze arguments from both sides, "
        "identify strengths and weaknesses, and provide balanced synthesis. "
        "You consider multiple perspectives including safety, innovation, economic impact, "
        "and ethical considerations. You may declare a winner or provide a refined "
        "answer that incorporates the best elements from both arguments, such as "
        "balanced regulatory frameworks that protect public interest while fostering innovation."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the DebateWithJudge system
debate_system = DebateWithJudge(
    pro_agent=pro_agent,
    con_agent=con_agent,
    judge_agent=judge_agent,
    max_rounds=3,
    output_type="str-all-except-first",
    verbose=True,
)

# Define the debate topic
topic = (
    "Should artificial intelligence be regulated by governments? "
    "Discuss the balance between innovation and safety, considering "
    "both the potential benefits of regulation (safety, ethics, public trust) "
    "and the potential drawbacks (stifling innovation, economic impact, "
    "regulatory capture). Provide a nuanced analysis."
)

# Run the debate
result = debate_system.run(task=topic)
print(result)

# Get the final refined answer
final_answer = debate_system.get_final_answer()
print(final_answer)
