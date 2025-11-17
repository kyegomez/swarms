import logging
from swarms.structs import Agent
from swarms.prompts.logistics import (
    Quality_Control_Agent_Prompt,
)

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)

# Image for analysis
# factory_image="image.png"   # normal image of a factory

factory_image = "image2.png"  # image of a burning factory


def security_analysis(danger_level: str) -> str:
    """
    Analyzes the security danger level and returns an appropriate response.

    Args:
        danger_level (str): The level of danger to analyze.
            Must be one of: "low", "medium", "high"

    Returns:
        str: A detailed security analysis based on the danger level.
    """
    if danger_level == "low":
        return """SECURITY ANALYSIS - LOW DANGER LEVEL:
        ✅ Environment appears safe and well-controlled
        ✅ Standard security measures are adequate
        ✅ Low risk of accidents or security breaches
        ✅ Normal operational protocols can continue
        
        Recommendations: Maintain current security standards and continue regular monitoring."""

    elif danger_level == "medium":
        return """SECURITY ANALYSIS - MEDIUM DANGER LEVEL:
        ⚠️  Moderate security concerns identified
        ⚠️  Enhanced monitoring recommended
        ⚠️  Some security measures may need strengthening
        ⚠️  Risk of incidents exists but manageable
        
        Recommendations: Implement additional safety protocols, increase surveillance, and conduct safety briefings."""

    elif danger_level == "high":
        return """SECURITY ANALYSIS - HIGH DANGER LEVEL:
        🚨 CRITICAL SECURITY CONCERNS DETECTED
        🚨 Immediate action required
        🚨 High risk of accidents or security breaches
        🚨 Operations may need to be suspended
        
        Recommendations: Immediate intervention required, evacuate if necessary, implement emergency protocols, and conduct thorough security review."""

    else:
        return f"ERROR: Invalid danger level '{danger_level}'. Must be 'low', 'medium', or 'high'."


# Custom system prompt that includes tool usage
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
    model_name="gpt-4.1",
    system_prompt=custom_system_prompt,
    multi_modal=True,
    max_loops=1,
    output_type="str-all-except-first",
    # tools_list_dictionary=[schema],
    tools=[security_analysis],
)


response = quality_control_agent.run(
    task="Analyze the image and then perform a security analysis. Based on what you see in the image, determine if there is a low, medium, or high danger level and call the security_analysis function with that danger level.",
    img=factory_image,
)

# The response is already printed by the agent's pretty_print method
