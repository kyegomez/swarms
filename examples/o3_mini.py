from swarms import Agent

Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="groq/deepseek-r1-distill-llama-70b",
    max_loops="auto",
    interactive=True,
    streaming_on=False,
).run("What are the best ways to analyze macroeconomic data?")
