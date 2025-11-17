from swarms import Agent


# Image for analysis
factory_image = "image.jpg"

# Quality control agent
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides a detailed report on the quality of the product in the image.",
    model_name="claude-3-5-sonnet-20240620",
    # system_prompt=Quality_Control_Agent_Prompt,
    # multi_modal=True,
    max_loops=1,
    output_type="str-all-except-first",
    summarize_multiple_images=True,
)


response = quality_control_agent.run(
    task="Analyze our factories images and provide a detailed health report for each factory.",
    imgs=[factory_image, "burning_image.jpg"],
)

print(response)
