"""
Example 3: Business Strategy Debate with Custom Configuration

This example demonstrates a business strategy debate with custom agent configurations,
multiple rounds, and accessing conversation history.
"""

from swarms import Agent, DebateWithJudge

# Create business strategy agents with detailed expertise
pro_agent = Agent(
    agent_name="Growth-Strategy-Pro",
    system_prompt=(
        "You are a business strategy consultant specializing in aggressive growth strategies. "
        "You argue in favor of rapid expansion, market penetration, and scaling. "
        "You present arguments focusing on first-mover advantages, market share capture, "
        "network effects, and competitive positioning. You use case studies from "
        "successful companies like Amazon, Uber, and Airbnb to support your position."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

con_agent = Agent(
    agent_name="Sustainable-Growth-Pro",
    system_prompt=(
        "You are a business strategy consultant specializing in sustainable, profitable growth. "
        "You argue against aggressive expansion in favor of measured, sustainable growth. "
        "You present counter-arguments focusing on profitability, unit economics, "
        "sustainable competitive advantages, and avoiding overextension. You identify "
        "weaknesses in 'growth at all costs' approaches and provide compelling alternatives "
        "based on companies like Apple, Microsoft, and Berkshire Hathaway."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

judge_agent = Agent(
    agent_name="Strategy-Judge",
    system_prompt=(
        "You are a seasoned business strategist and former CEO evaluating growth strategy debates. "
        "You carefully analyze arguments from both sides, considering factors like: "
        "- Market conditions and competitive landscape\n"
        "- Company resources and capabilities\n"
        "- Risk tolerance and financial position\n"
        "- Long-term sustainability vs. short-term growth\n"
        "- Industry-specific dynamics\n\n"
        "You provide balanced synthesis that incorporates the best elements from both arguments, "
        "considering context-specific factors. You may recommend a hybrid approach when appropriate."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the debate system with extended loops for complex strategy discussions
strategy_debate = DebateWithJudge(
    pro_agent=pro_agent,
    con_agent=con_agent,
    judge_agent=judge_agent,
    max_loops=4,  # More loops for complex strategic discussions
    output_type="dict",  # Use dict format for structured analysis
    verbose=True,
)

# Define a complex business strategy question
strategy_question = (
    "A SaaS startup with $2M ARR, 40% gross margins, and $500K in the bank "
    "is considering two paths:\n"
    "1. Aggressive growth: Raise $10M, hire 50 people, expand to 5 new markets\n"
    "2. Sustainable growth: Focus on profitability, improve unit economics, "
    "expand gradually with existing resources\n\n"
    "Which strategy should they pursue? Consider market conditions, competitive "
    "landscape, and long-term viability."
)

# Run the debate
result = strategy_debate.run(task=strategy_question)
print(result)

# Get the full conversation history for detailed analysis
history = strategy_debate.get_conversation_history()
print(history)

# Get the final refined answer
final_answer = strategy_debate.get_final_answer()
print(final_answer)
