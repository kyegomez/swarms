"""
Sequential Multi-Agent Example

This example demonstrates sequential execution where agents work one after another,
with each agent building on the previous agent's output.

Use Case: Content creation pipeline where research → writing → editing happens in sequence
"""

from swarms import Agent

# Create specialized agents for sequential workflow
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert at researching topics and gathering information",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a research specialist. Gather comprehensive information on the given topic and provide key findings.",
)

writer_agent = Agent(
    agent_name="Writer-Agent",
    agent_description="Expert at writing engaging content based on research",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a professional writer. Transform research findings into engaging, well-structured content.",
)

editor_agent = Agent(
    agent_name="Editor-Agent",
    agent_description="Expert at editing and refining content for quality",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are an experienced editor. Review and improve the content for clarity, grammar, and impact.",
)

# Sequential execution: Research -> Write -> Edit
if __name__ == "__main__":
    topic = "The impact of artificial intelligence on healthcare"
    
    print("="*80)
    print("SEQUENTIAL AGENT WORKFLOW")
    print("="*80)
    
    # Step 1: Research
    print("\n[Step 1/3] Research Agent working...")
    research_output = research_agent.run(
        f"Research the following topic and provide key findings: {topic}"
    )
    print(f"Research completed: {len(research_output)} characters")
    
    # Step 2: Write
    print("\n[Step 2/3] Writer Agent working...")
    writer_output = writer_agent.run(
        f"Based on this research, write a comprehensive article:\n\n{research_output}"
    )
    print(f"Writing completed: {len(writer_output)} characters")
    
    # Step 3: Edit
    print("\n[Step 3/3] Editor Agent working...")
    final_output = editor_agent.run(
        f"Edit and improve this article for publication:\n\n{writer_output}"
    )
    
    # Display final result
    print("\n" + "="*80)
    print("FINAL ARTICLE")
    print("="*80)
    print(final_output)
    print("\n" + "="*80)
    print("Sequential workflow completed!")
    print("="*80)
