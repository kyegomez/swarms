from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent--new",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks and your name is Quantitative-Trading-Agent",
    model_name="openrouter/minimax/minimax-m3",
    max_loops=1,
    top_p=None,
    temperature=1.0,
    thinking_tokens=1024,
    tools_list_dictionary=None,
    reasoning_effort="high",
    persistent_memory=False,
)

out = agent.run(
    task="Analyze the best semiconductor ETFs and provide a detailed comparison. Include metrics such as performance, expense ratio, holdings, and any notable strategies.",
)

print(out)
