from swarms import Agent
from swarms.structs.groupchat import GroupChat, round_robin_speaker


def create_collaborative_agents():
    """Create agents designed for enhanced collaboration."""

    # Data Analyst - focuses on data insights and trends
    analyst = Agent(
        agent_name="analyst",
        system_prompt="""You are a senior data analyst with expertise in business intelligence, statistical analysis, and data visualization. You excel at:
- Analyzing complex datasets and identifying trends
- Creating actionable insights from data
- Providing quantitative evidence for business decisions
- Identifying patterns and correlations in data

When collaborating, always reference specific data points and build upon others' insights with quantitative support.""",
        llm="gpt-3.5-turbo",
    )

    # Market Researcher - focuses on market trends and customer insights
    researcher = Agent(
        agent_name="researcher",
        system_prompt="""You are a market research specialist with deep expertise in consumer behavior, competitive analysis, and market trends. You excel at:
- Understanding customer needs and preferences
- Analyzing competitive landscapes
- Identifying market opportunities and threats
- Providing qualitative insights that complement data analysis

When collaborating, always connect market insights to business implications and build upon data analysis with market context.""",
        llm="gpt-3.5-turbo",
    )

    # Strategy Consultant - focuses on strategic recommendations
    strategist = Agent(
        agent_name="strategist",
        system_prompt="""You are a strategic consultant with expertise in business strategy, competitive positioning, and strategic planning. You excel at:
- Developing comprehensive business strategies
- Identifying competitive advantages
- Creating actionable strategic recommendations
- Synthesizing multiple perspectives into coherent strategies

When collaborating, always synthesize insights from all team members and provide strategic recommendations that leverage the collective expertise.""",
        llm="gpt-3.5-turbo",
    )

    return [analyst, researcher, strategist]


def example_comprehensive_analysis():
    """Example of comprehensive collaborative analysis."""
    print("=== Enhanced Collaborative Analysis Example ===\n")

    agents = create_collaborative_agents()

    # Create group chat with round robin speaker function
    group_chat = GroupChat(
        name="Strategic Analysis Team",
        description="A collaborative team for comprehensive business analysis",
        agents=agents,
        speaker_function=round_robin_speaker,
        interactive=False,
    )

    # Complex task that requires collaboration
    task = """Analyze our company's performance in the e-commerce market. 
    We have the following data:
    - Q3 revenue: $2.5M (up 15% from Q2)
    - Customer acquisition cost: $45 (down 8% from Q2)
    - Customer lifetime value: $180 (up 12% from Q2)
    - Market share: 3.2% (up 0.5% from Q2)
    - Competitor analysis shows 3 major players with 60% market share combined
    
    @analyst @researcher @strategist please provide a comprehensive analysis and strategic recommendations."""

    print(f"Task: {task}\n")
    print("Expected collaborative behavior:")
    print(
        "1. Analyst: Analyzes the data trends and provides quantitative insights"
    )
    print(
        "2. Researcher: Builds on data with market context and competitive analysis"
    )
    print(
        "3. Strategist: Synthesizes both perspectives into strategic recommendations"
    )
    print("\n" + "=" * 80 + "\n")

    response = group_chat.run(task)
    print(f"Collaborative Response:\n{response}")


def example_problem_solving():
    """Example of collaborative problem solving."""
    print("\n" + "=" * 80)
    print("=== Collaborative Problem Solving Example ===\n")

    agents = create_collaborative_agents()

    group_chat = GroupChat(
        name="Problem Solving Team",
        description="A team that collaborates to solve complex business problems",
        agents=agents,
        speaker_function=round_robin_speaker,
        interactive=False,
    )

    # Problem-solving task
    task = """We're experiencing declining customer retention rates (down 20% in the last 6 months). 
    Our customer satisfaction scores are also dropping (from 8.5 to 7.2).
    
    @analyst please analyze the retention data, @researcher investigate customer feedback and market trends, 
    and @strategist develop a comprehensive solution strategy."""

    print(f"Task: {task}\n")
    print("Expected collaborative behavior:")
    print("1. Analyst: Identifies patterns in retention data")
    print(
        "2. Researcher: Explores customer feedback and market factors"
    )
    print(
        "3. Strategist: Combines insights to create actionable solutions"
    )
    print("\n" + "=" * 80 + "\n")

    response = group_chat.run(task)
    print(f"Collaborative Response:\n{response}")


def example_agent_delegation():
    """Example showing how agents delegate to each other."""
    print("\n" + "=" * 80)
    print("=== Agent Delegation Example ===\n")

    agents = create_collaborative_agents()

    group_chat = GroupChat(
        name="Delegation Team",
        description="A team that demonstrates effective delegation and collaboration",
        agents=agents,
        speaker_function=round_robin_speaker,
        interactive=False,
    )

    # Task that encourages delegation
    task = """We need to evaluate a potential new market entry opportunity in Southeast Asia.
    The initial data shows promising growth potential, but we need a comprehensive assessment.
    
    @analyst start with the market data analysis, then delegate to @researcher for market research, 
    and finally @strategist should provide strategic recommendations."""

    print(f"Task: {task}\n")
    print("Expected behavior:")
    print(
        "1. Analyst: Analyzes data and delegates to researcher for deeper market insights"
    )
    print(
        "2. Researcher: Builds on data analysis and delegates to strategist for recommendations"
    )
    print(
        "3. Strategist: Synthesizes all insights into strategic recommendations"
    )
    print("\n" + "=" * 80 + "\n")

    response = group_chat.run(task)
    print(f"Collaborative Response:\n{response}")


def example_synthesis_and_integration():
    """Example showing synthesis of multiple perspectives."""
    print("\n" + "=" * 80)
    print("=== Synthesis and Integration Example ===\n")

    agents = create_collaborative_agents()

    group_chat = GroupChat(
        name="Synthesis Team",
        description="A team that excels at integrating multiple perspectives",
        agents=agents,
        speaker_function=round_robin_speaker,
        interactive=False,
    )

    # Task requiring synthesis
    task = """We have conflicting information about our product's market position:
    - Sales data shows strong growth (25% increase)
    - Customer surveys indicate declining satisfaction
    - Competitor analysis shows we're losing market share
    - Internal metrics show improved operational efficiency
    
    @analyst @researcher @strategist please analyze these conflicting signals and provide 
    an integrated assessment of our true market position."""

    print(f"Task: {task}\n")
    print("Expected behavior:")
    print(
        "1. Analyst: Clarifies the data discrepancies and identifies patterns"
    )
    print(
        "2. Researcher: Provides market context to explain the contradictions"
    )
    print(
        "3. Strategist: Synthesizes all perspectives into a coherent market assessment"
    )
    print("\n" + "=" * 80 + "\n")

    response = group_chat.run(task)
    print(f"Collaborative Response:\n{response}")


def main():
    """Run all enhanced collaboration examples."""
    print("Enhanced Collaborative GroupChat Examples")
    print("=" * 80)
    print("This demonstrates improved agent collaboration with:")
    print("- Acknowledgment of other agents' contributions")
    print("- Building upon previous insights")
    print("- Synthesis of multiple perspectives")
    print("- Appropriate delegation using @mentions")
    print("- Comprehensive understanding of conversation history")
    print("=" * 80 + "\n")

    # Run examples
    example_comprehensive_analysis()
    example_problem_solving()
    example_agent_delegation()
    example_synthesis_and_integration()

    print("\n" + "=" * 80)
    print("All enhanced collaboration examples completed!")
    print("Notice how agents now:")
    print("✓ Acknowledge each other's contributions")
    print("✓ Build upon previous insights")
    print("✓ Synthesize multiple perspectives")
    print("✓ Delegate appropriately")
    print("✓ Provide more cohesive and comprehensive responses")


if __name__ == "__main__":
    main()
