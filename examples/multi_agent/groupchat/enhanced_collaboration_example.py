"""Enhanced collaboration examples using the dynamic GroupChat.

Each scenario seeds the same team of analyst / researcher / strategist agents
with a different business prompt and lets them self-select who replies.
"""

from swarms import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL


def create_collaborative_agents():
    """Create agents designed for enhanced collaboration."""

    analyst = Agent(
        agent_name="analyst",
        system_prompt="""You are a senior data analyst with expertise in business intelligence, statistical analysis, and data visualization. You excel at:
- Analyzing complex datasets and identifying trends
- Creating actionable insights from data
- Providing quantitative evidence for business decisions
- Identifying patterns and correlations in data

When collaborating, always reference specific data points and build upon others' insights with quantitative support.""",
        model_name="gpt-4.1",
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

    researcher = Agent(
        agent_name="researcher",
        system_prompt="""You are a market research specialist with deep expertise in consumer behavior, competitive analysis, and market trends. You excel at:
- Understanding customer needs and preferences
- Analyzing competitive landscapes
- Identifying market opportunities and threats
- Providing qualitative insights that complement data analysis

When collaborating, always connect market insights to business implications and build upon data analysis with market context.""",
        model_name="gpt-4.1",
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

    strategist = Agent(
        agent_name="strategist",
        system_prompt="""You are a strategic consultant with expertise in business strategy, competitive positioning, and strategic planning. You excel at:
- Developing comprehensive business strategies
- Identifying competitive advantages
- Creating actionable strategic recommendations
- Synthesizing multiple perspectives into coherent strategies

When collaborating, always synthesize insights from all team members and provide strategic recommendations that leverage the collective expertise.""",
        model_name="gpt-4.1",
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )

    return [analyst, researcher, strategist]


def _make_chat(name: str, description: str):
    return GroupChat(
        name=name,
        description=description,
        agents=create_collaborative_agents(),
        max_loops=12,
        threshold=0.5,
        idle_timeout=8.0,
    )


def example_comprehensive_analysis():
    """Example of comprehensive collaborative analysis."""
    print("=== Enhanced Collaborative Analysis Example ===\n")

    chat = _make_chat(
        "Strategic Analysis Team",
        "A collaborative team for comprehensive business analysis",
    )

    task = """Analyze our company's performance in the e-commerce market.
    We have the following data:
    - Q3 revenue: $2.5M (up 15% from Q2)
    - Customer acquisition cost: $45 (down 8% from Q2)
    - Customer lifetime value: $180 (up 12% from Q2)
    - Market share: 3.2% (up 0.5% from Q2)
    - Competitor analysis shows 3 major players with 60% market share combined

    @analyst @researcher @strategist please provide a comprehensive analysis and strategic recommendations."""

    print(f"Task: {task}\n")
    print("=" * 80 + "\n")

    response = chat.run(task)
    print(f"Collaborative Response:\n{response}")


def example_problem_solving():
    """Example of collaborative problem solving."""
    print("\n" + "=" * 80)
    print("=== Collaborative Problem Solving Example ===\n")

    chat = _make_chat(
        "Problem Solving Team",
        "A team that collaborates to solve complex business problems",
    )

    task = """We're experiencing declining customer retention rates (down 20% in the last 6 months).
    Our customer satisfaction scores are also dropping (from 8.5 to 7.2).

    @analyst please analyze the retention data, @researcher investigate customer feedback and market trends,
    and @strategist develop a comprehensive solution strategy."""

    print(f"Task: {task}\n")
    print("=" * 80 + "\n")

    response = chat.run(task)
    print(f"Collaborative Response:\n{response}")


def example_agent_delegation():
    """Example showing how agents delegate to each other."""
    print("\n" + "=" * 80)
    print("=== Agent Delegation Example ===\n")

    chat = _make_chat(
        "Delegation Team",
        "A team that demonstrates effective delegation and collaboration",
    )

    task = """We need to evaluate a potential new market entry opportunity in Southeast Asia.
    The initial data shows promising growth potential, but we need a comprehensive assessment.

    @analyst start with the market data analysis, then @researcher add deeper market insights,
    and finally @strategist should provide strategic recommendations."""

    print(f"Task: {task}\n")
    print("=" * 80 + "\n")

    response = chat.run(task)
    print(f"Collaborative Response:\n{response}")


def example_synthesis_and_integration():
    """Example showing synthesis of multiple perspectives."""
    print("\n" + "=" * 80)
    print("=== Synthesis and Integration Example ===\n")

    chat = _make_chat(
        "Synthesis Team",
        "A team that excels at integrating multiple perspectives",
    )

    task = """We have conflicting information about our product's market position:
    - Sales data shows strong growth (25% increase)
    - Customer surveys indicate declining satisfaction
    - Competitor analysis shows we're losing market share
    - Internal metrics show improved operational efficiency

    @analyst @researcher @strategist please analyze these conflicting signals and provide
    an integrated assessment of our true market position."""

    print(f"Task: {task}\n")
    print("=" * 80 + "\n")

    response = chat.run(task)
    print(f"Collaborative Response:\n{response}")


def main():
    """Run all enhanced collaboration examples."""
    print("Enhanced Collaborative GroupChat Examples")
    print("=" * 80 + "\n")

    example_comprehensive_analysis()
    example_problem_solving()
    example_agent_delegation()
    example_synthesis_and_integration()

    print("\n" + "=" * 80)
    print("All enhanced collaboration examples completed!")


if __name__ == "__main__":
    main()
