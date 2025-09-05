from swarms import Agent


SYSTEM_PROMPT = (
    "You are an expert in image geolocalization. Given an image, provide the most likely location it was taken. "
    "Analyze visual cues such as architecture, landscape, vegetation, weather patterns, cultural elements, "
    "and any other geographical indicators to determine the precise location. Provide your reasoning and "
    "confidence level for your prediction."
)

# Agent for image geolocalization
agent = Agent(
    agent_name="Geo-Guesser-Agent",
    agent_description="Expert agent specialized in image geolocalization, capable of identifying geographical locations from visual cues in images.",
    model_name="gemini/gemini-2.5-flash-image-preview",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    retry_interval=1,
)

out = agent.run(
    task=f"{SYSTEM_PROMPT}",
    img="miami.jpg",
)

print(out)
