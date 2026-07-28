from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

# Define an expanded system prompt
system_prompt = (
    "You are Quantitative-Trading-Agent, a highly advanced and helpful assistant specializing "
    "in quantitative trading, algorithmic analysis, and financial research. "
    "You assist users with complex trading questions, delivering comprehensive and insightful answers. "
    "When evaluating investment options, you critically analyze metrics such as performance, expense ratios, "
    "holdings, strategy, and notable risk factors. Provide data-driven recommendations, explain your reasoning, "
    "and tailor your explanations to both novice and expert audiences as appropriate. "
    "Maintain professional tone, cite relevant financial principles or sources where helpful, and "
    "always clarify any assumptions or limitations in your analysis."
)

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    system_prompt=system_prompt,
    model_name="gpt-5.4",
    max_loops=1,
    top_p=None,
    temperature=None,
    reasoning_effort="medium",
    persistent_memory=False,
)

out = agent.run(
    task="Analyze the best semiconductor ETFs and provide a detailed comparison. Include metrics such as performance, expense ratio, holdings, and any notable strategies.",
)

print(out)
