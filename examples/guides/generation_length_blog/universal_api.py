from swarms import Agent


def generate_comprehensive_content(topic, sections):
    prompt = f"""You are tasked with creating a comprehensive, detailed analysis of {topic}. 
    This should be a thorough, professional-level document suitable for expert review.
    
    Structure your response with the following sections, ensuring each is substantive and detailed:
    {chr(10).join([f"{i+1}. {section} - Provide extensive detail with examples and analysis" for i, section in enumerate(sections)])}
    
    For each section:
    - Include multiple subsections where appropriate
    - Provide specific examples and case studies
    - Offer detailed explanations of complex concepts
    - Include relevant technical details and specifications
    - Discuss implications and considerations thoroughly
    
    Aim for comprehensive coverage that demonstrates deep expertise. This is a professional document that should be thorough and substantive throughout."""

    agent = Agent(
        name="Comprehensive Content Generator",
        system_prompt=prompt,
        model="claude-sonnet-4-20250514",
        max_loops=1,
        temperature=0.5,
        max_tokens=4000,
    )

    return agent.run(topic)
