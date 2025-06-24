from swarms.structs import Agent
from swarms.prompts.logistics import (
    Quality_Control_Agent_Prompt,
)


# Image for analysis
factory_image = "image.jpg"


def security_analysis(danger_level: str = None) -> str:
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


# schema = BaseTool().function_to_dict(security_analysis)
# print(json.dumps(schema, indent=4))

# Quality control agent
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides a detailed report on the quality of the product in the image.",
    # model_name="anthropic/claude-3-opus-20240229",
    model_name="gpt-4o-mini",
    system_prompt=Quality_Control_Agent_Prompt,
    multi_modal=True,
    max_loops=1,
    output_type="str-all-except-first",
    # tools_list_dictionary=[schema],
    tools=[security_analysis],
)


response = quality_control_agent.run(
    task="what is in the image?",
    # img=factory_image,
)

print(response)
