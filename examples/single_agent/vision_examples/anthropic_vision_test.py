from swarms.structs import Agent
from swarms.prompts.logistics import (
    Quality_Control_Agent_Prompt,
)

# Image for analysis
factory_image = "image.jpg"


# Quality control agent
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides a detailed report on the quality of the product in the image.",
    model_name="anthropic/claude-3-opus-20240229",
    system_prompt=Quality_Control_Agent_Prompt,
    multi_modal=True,
    max_loops=1,
    output_type="str-all-except-first",
)

response = quality_control_agent.run(
    task="Create a comprehensive report on the factory image and it's status",
    img=factory_image,
)

print(response)
