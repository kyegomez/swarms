from swarms import Agent


SYSTEM_PROMPT = (
    "You are a location-based AR experience generator. Highlight points of interest in this image and annotate relevant information about it. "
    "Generate the new image only."
)

# Agent for AR annotation
agent = Agent(
    agent_name="Tactical-Strategist-Agent",
    agent_description="Agent specialized in tactical strategy, scenario analysis, and actionable recommendations for complex situations.",
    model_name="gemini/gemini-2.5-flash-image-preview", #"gemini/gemini-2.5-flash-image-preview",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    retry_interval=1,
    verbose=True,
)


out = agent.run(
    task=f"{SYSTEM_PROMPT} \n\n Annotate all the tallest buildings in the image",
    img="hk.jpg",
)

