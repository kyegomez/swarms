from swarms import Agent

# SYSTEM_PROMPT = (
#     "You are an expert system for generating immersive, location-based augmented reality (AR) experiences. "
#     "Given an input image, your task is to thoroughly analyze the scene and identify every point of interest (POI), "
#     "including landmarks, objects, architectural features, signage, and any elements relevant to the location or context. "
#     "For each POI you detect, provide a clear annotation that includes:\n"
#     "- A concise label or title for the POI\n"
#     "- A detailed description explaining its significance, historical or cultural context, or practical information\n"
#     "- Any relevant facts, trivia, or actionable insights that would enhance a user's AR experience\n"
#     "Present your output as a structured list, with each POI clearly separated. "
#     "Be thorough, accurate, and engaging, ensuring that your annotations would be valuable for users exploring the location through AR. "
#     "If possible, infer connections between POIs and suggest interactive or educational opportunities."
#     "Do not provide any text, annotation, or explanation—simply output the generated or processed image as your response."
# )


SYSTEM_PROMPT = (
    "You are a location-based AR experience generator. Highlight points of interest in this image and annotate relevant information about it. "
    "Return the image only."
)

agent = Agent(
    agent_name="Tactical-Strategist-Agent",
    agent_description="Agent specialized in tactical strategy, scenario analysis, and actionable recommendations for complex situations.",
    model_name="gemini/gemini-2.5-flash-image-preview",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    retry_interval=1,
)

out = agent.run(
    task=f"{SYSTEM_PROMPT} \n\n Annotate all the tallest buildings in the image",
    img="hk.jpg",
)

print(out)
