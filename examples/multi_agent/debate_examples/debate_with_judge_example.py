from swarms import DebateWithJudge

debate_system = DebateWithJudge(
    preset_agents=True,
    max_loops=3,
    model_name="gpt-4o-mini",
    output_type="str-all-except-first",
    verbose=True,
)

topic = (
    "Should artificial intelligence be regulated by governments? "
    "Discuss the balance between innovation and safety."
)

result = debate_system.run(task=topic)
