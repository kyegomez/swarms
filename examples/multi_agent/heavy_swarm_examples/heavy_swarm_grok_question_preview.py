"""
Grok 4.20 Heavy mode — question preview.

Preview what Captain Swarm will ask each specialist
without running the full pipeline.
"""

from swarms import HeavySwarm

swarm = HeavySwarm(
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    variant="medium",
)

task = "Should we migrate our monolith to microservices?"

# Get questions as a dictionary
questions = swarm.get_questions_only(task)
print("=== Questions by Agent ===")
print(f"Harper (facts):    {questions['harper_question']}")
print(f"Benjamin (logic):  {questions['benjamin_question']}")
print(f"Lucas (creative):  {questions['lucas_question']}")

# Or as an ordered list
print("\n=== Questions as List ===")
question_list = swarm.get_questions_as_list(task)
agents = ["Harper", "Benjamin", "Lucas"]
for name, q in zip(agents, question_list):
    print(f"  {name}: {q}")
