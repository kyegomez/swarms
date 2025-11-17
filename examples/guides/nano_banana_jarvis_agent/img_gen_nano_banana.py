from swarms import Agent

IMAGE_GEN_SYSTEM_PROMPT = (
    "You are an advanced image generation agent. Given a textual description, generate a high-quality, photorealistic image that matches the prompt. "
    "Return only the generated image."
)

image_gen_agent = Agent(
    agent_name="Image-Generation-Agent",
    agent_description="Agent specialized in generating high-quality, photorealistic images from textual prompts.",
    model_name="gemini/gemini-2.5-flash-image-preview",  # Replace with your preferred image generation model if available
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    retry_interval=1,
)

image_gen_out = image_gen_agent.run(
    task=f"{IMAGE_GEN_SYSTEM_PROMPT} \n\n Generate a photorealistic image of a futuristic city skyline at sunset.",
)

print("Image Generation Output:")
print(image_gen_out)
