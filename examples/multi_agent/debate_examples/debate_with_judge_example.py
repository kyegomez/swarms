from swarms import Agent, DebateWithJudge

# Create the Pro agent (arguing in favor)
pro_agent = Agent(
    agent_name="Pro-Agent",
    system_prompt=(
        "You are a skilled debater who argues in favor of positions. "
        "You present well-reasoned arguments with evidence, examples, "
        "and logical reasoning. You are persuasive and articulate."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the Con agent (arguing against)
con_agent = Agent(
    agent_name="Con-Agent",
    system_prompt=(
        "You are a skilled debater who argues against positions. "
        "You present strong counter-arguments with evidence, examples, "
        "and logical reasoning. You identify weaknesses in opposing "
        "arguments and provide compelling alternatives."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the Judge agent (evaluates and synthesizes)
judge_agent = Agent(
    agent_name="Judge-Agent",
    system_prompt=(
        "You are an impartial judge who evaluates debates. "
        "You carefully analyze arguments from both sides, identify "
        "strengths and weaknesses, and provide balanced synthesis. "
        "You may declare a winner or provide a refined answer that "
        "incorporates the best elements from both arguments."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the DebateWithJudge system
debate_system = DebateWithJudge(
    pro_agent=pro_agent,
    con_agent=con_agent,
    judge_agent=judge_agent,
    max_rounds=3,  # Run 3 rounds of debate and refinement
    output_type="str-all-except-first",  # Return as formatted string
    verbose=True,  # Enable verbose logging
)

# Define the debate topic
topic = (
    "Should artificial intelligence be regulated by governments? "
    "Discuss the balance between innovation and safety."
)

# Run the debate
result = debate_system.run(task=topic)

print(result)
