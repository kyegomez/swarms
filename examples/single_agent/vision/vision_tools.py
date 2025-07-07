from swarms.structs import Agent
from swarms.prompts.logistics import (
    Quality_Control_Agent_Prompt,
)


# Image for analysis
factory_image = "image.jpg"


def security_analysis(danger_level: str) -> str:
    """
    Analyzes the security danger level and returns an appropriate response.

    Args:
        danger_level (str, optional): The level of danger to analyze.
            Can be "low", "medium", "high", or None. Defaults to None.

    Returns:
        str: A string describing the danger level assessment.
            - "No danger level provided" if danger_level is None
            - "No danger" if danger_level is "low"
            - "Medium danger" if danger_level is "medium"
            - "High danger" if danger_level is "high"
            - "Unknown danger level" for any other value
    """
    if danger_level is None:
        return "No danger level provided"

    if danger_level == "low":
        return "No danger"

    if danger_level == "medium":
        return "Medium danger"

    if danger_level == "high":
        return "High danger"

    return "Unknown danger level"


custom_system_prompt = f"""
{Quality_Control_Agent_Prompt}

You have access to tools that can help you with your analysis. When you need to perform a security analysis, you MUST use the security_analysis function with an appropriate danger level (low, medium, or high) based on your observations.

Always use the available tools when they are relevant to the task. If you determine there is any level of danger or security concern, call the security_analysis function with the appropriate danger level.
"""

# Quality control agent
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides a detailed report on the quality of the product in the image.",
    # model_name="anthropic/claude-3-opus-20240229",
    model_name="gpt-4o-mini",
    system_prompt=custom_system_prompt,
    multi_modal=True,
    max_loops=1,
    output_type="str-all-except-first",
    # tools_list_dictionary=[schema],
    tools=[security_analysis],
)


response = quality_control_agent.run(
    task="Analyze the image and then perform a security analysis. Based on what you see in the image, determine if there is a low, medium, or high danger level and call the security_analysis function with that danger level",
    img=factory_image,
)
