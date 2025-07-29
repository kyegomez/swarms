"""
Swarms Conversation Supabase Backend Example

This example demonstrates using Supabase as the conversation storage backend
with real Swarms agents. It shows how to:
1. Set up agents with different roles and capabilities
2. Use Supabase for persistent conversation storage
3. Aggregate multi-agent responses
4. Handle real-world agent workflows

Prerequisites:
- Supabase project with SUPABASE_URL and SUPABASE_ANON_KEY environment variables
- LLM API keys (OpenAI, Anthropic, etc.)
- pip install supabase swarms
"""

import os
from typing import List
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.utils.history_output_formatter import (
    history_output_formatter,
    HistoryOutputType,
)


def aggregator_agent_task_prompt(
    task: str, workers: List[Agent], conversation: Conversation
):
    """Create a comprehensive prompt for the aggregator agent."""
    return f"""
    As an expert analysis aggregator, please synthesize the following multi-agent conversation 
    into a comprehensive and actionable report.

    Original Task: {task}
    Number of Participating Agents: {len(workers)}
    Agent Roles: {', '.join([agent.agent_name for agent in workers])}

    Conversation Content:
    {conversation.get_str()}

    Please provide a detailed synthesis that includes:
    1. Executive Summary
    2. Key Insights from each agent
    3. Conflicting viewpoints (if any)
    4. Recommended actions
    5. Next steps

    Format your response as a professional report.
    """


def create_research_agents() -> List[Agent]:
    """Create a team of specialized research agents."""

    # Data Analyst Agent
    data_analyst = Agent(
        agent_name="DataAnalyst",
        agent_description="Expert in data analysis, statistics, and market research",
        system_prompt="""You are a senior data analyst with expertise in:
        - Statistical analysis and data interpretation
        - Market research and trend analysis
        - Data visualization insights
        - Quantitative research methods
        
        Provide data-driven insights with specific metrics, trends, and statistical evidence.
        Always cite data sources and provide confidence levels for your analysis.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=True,
        output_type="string",
    )

    # Research Specialist Agent
    researcher = Agent(
        agent_name="ResearchSpecialist",
        agent_description="Expert in academic research, industry analysis, and information synthesis",
        system_prompt="""You are a research specialist with expertise in:
        - Academic research and literature review
        - Industry analysis and competitive intelligence
        - Information synthesis and validation
        - Research methodology and best practices
        
        Provide well-researched, evidence-based insights with proper citations.
        Focus on credible sources and peer-reviewed information when possible.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=True,
        output_type="string",
    )

    # Strategic Advisor Agent
    strategist = Agent(
        agent_name="StrategicAdvisor",
        agent_description="Expert in strategic planning, business strategy, and decision-making",
        system_prompt="""You are a strategic advisor with expertise in:
        - Strategic planning and business strategy
        - Risk assessment and mitigation
        - Decision-making frameworks
        - Competitive analysis and positioning
        
        Provide strategic recommendations with clear rationale.
        Focus on actionable insights and long-term implications.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=True,
        output_type="string",
    )

    return [data_analyst, researcher, strategist]


def create_aggregator_agent() -> Agent:
    """Create an aggregator agent to synthesize multi-agent responses."""
    return Agent(
        agent_name="SynthesisAggregator",
        agent_description="Expert in analyzing and synthesizing multi-agent conversations into comprehensive reports",
        system_prompt="""You are an expert synthesis aggregator specializing in:
        - Multi-perspective analysis and integration
        - Comprehensive report generation
        - Conflict resolution and consensus building
        - Strategic insight extraction
        
        Your role is to analyze conversations between multiple expert agents and create 
        a unified, actionable report that captures the best insights from all participants.
        
        Always structure your reports professionally with:
        - Executive Summary
        - Detailed Analysis
        - Key Recommendations
        - Implementation Steps""",
        model_name="gpt-4o",
        max_loops=1,
        verbose=True,
        output_type="string",
    )


def aggregate_with_supabase(
    workers: List[Agent],
    task: str = None,
    type: HistoryOutputType = "all",
    aggregator_model_name: str = "gpt-4o",
    # Backend parameters for conversation storage
    backend: str = "supabase",
    supabase_url: str = None,
    supabase_key: str = None,
):
    """
    Aggregate agent responses using Supabase for conversation storage.

    Args:
        workers: List of Agent instances
        task: The task to execute
        type: Output type for history formatting
        aggregator_model_name: Model name for the aggregator agent
        backend: Storage backend (default: "supabase")
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
    """

    if task is None:
        raise ValueError("Task is required for agent aggregation")

    if not workers:
        raise ValueError("At least one worker agent is required")

    if not all(isinstance(worker, Agent) for worker in workers):
        raise ValueError("All workers must be Agent instances")

    # Set up Supabase conversation storage
    conversation_kwargs = {}
    if backend == "supabase":
        url = supabase_url or os.getenv("SUPABASE_URL")
        key = supabase_key or os.getenv("SUPABASE_ANON_KEY")

        if not url or not key:
            raise ValueError(
                "Supabase backend requires SUPABASE_URL and SUPABASE_ANON_KEY "
                "environment variables or explicit parameters"
            )

        conversation_kwargs.update(
            {
                "supabase_url": url,
                "supabase_key": key,
            }
        )

    try:
        # Create conversation with Supabase backend
        conversation = Conversation(
            backend=backend,
            **conversation_kwargs,
            system_prompt="Multi-agent collaboration session with persistent storage",
            time_enabled=True,
        )
        print(
            f"✅ Successfully initialized {backend} backend for conversation storage"
        )

        # Add initial task to conversation
        conversation.add("system", f"Task: {task}")

    except ImportError as e:
        print(f"❌ Backend initialization failed: {e}")
        print("💡 Falling back to in-memory storage")
        conversation = Conversation(backend="in-memory")

    # Create aggregator agent
    aggregator_agent = create_aggregator_agent()

    print(
        f"🚀 Starting multi-agent execution with {len(workers)} agents..."
    )

    # Run agents concurrently
    results = run_agents_concurrently(agents=workers, task=task)

    # Store individual agent responses in conversation
    for result, agent in zip(results, workers):
        conversation.add(content=result, role=agent.agent_name)
        print(f"📝 Stored response from {agent.agent_name}")

    print("🔄 Running aggregation analysis...")

    # Generate aggregated analysis
    final_result = aggregator_agent.run(
        task=aggregator_agent_task_prompt(task, workers, conversation)
    )

    # Store aggregated result
    conversation.add(
        content=final_result, role=aggregator_agent.agent_name
    )

    print("✅ Aggregation complete!")

    # Return formatted history
    return history_output_formatter(
        conversation=conversation, type=type
    )


# Example usage with real Swarms agents
if __name__ == "__main__":
    print(
        "🧪 Testing Swarms Multi-Agent System with Supabase Backend"
    )
    print("=" * 70)

    # Check environment setup
    print("\n⚙️  Environment Setup Check")
    print("-" * 40)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print(
        f"SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Not set'}"
    )
    print(
        f"SUPABASE_ANON_KEY: {'✅ Set' if supabase_key else '❌ Not set'}"
    )
    print(
        f"OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}"
    )

    if not (supabase_url and supabase_key):
        print("\n⚠️  Missing Supabase configuration!")
        print("Please set the following environment variables:")
        print("export SUPABASE_URL=https://your-project.supabase.co")
        print("export SUPABASE_ANON_KEY=your-anon-key")
        print("\nFalling back to demonstration with mock data...")

    if not openai_key:
        print("\n⚠️  Missing OpenAI API key!")
        print("Please set: export OPENAI_API_KEY=your-api-key")
        print(
            "You can also use other LLM providers (Anthropic, Google, etc.)"
        )

    # Example 1: Basic Multi-Agent Research Task
    print("\n📦 Example 1: Multi-Agent Market Research")
    print("-" * 50)

    try:
        # Create research team
        research_team = create_research_agents()

        # Define research task
        research_task = """
        Analyze the current state and future prospects of artificial intelligence 
        in healthcare. Consider market trends, technological developments, 
        regulatory challenges, and investment opportunities. Provide insights 
        on key players, emerging technologies, and potential risks.
        """

        print(f"📋 Task: {research_task.strip()}")
        print(
            f"👥 Team: {[agent.agent_name for agent in research_team]}"
        )

        if supabase_url and supabase_key and openai_key:
            # Run with real agents and Supabase storage
            result = aggregate_with_supabase(
                workers=research_team,
                task=research_task,
                type="final",
                backend="supabase",
                supabase_url=supabase_url,
                supabase_key=supabase_key,
            )

            print("\n📊 Research Results:")
            print("=" * 50)
            print(result)

        else:
            print(
                "❌ Skipping real agent execution due to missing configuration"
            )

    except Exception as e:
        print(f"❌ Error in multi-agent research: {e}")

    # Example 2: Simple Conversation Storage Test
    print("\n📦 Example 2: Direct Conversation Storage Test")
    print("-" * 50)

    try:
        if supabase_url and supabase_key:
            # Test direct conversation with Supabase
            conv = Conversation(
                backend="supabase",
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                time_enabled=True,
            )

            print("✅ Supabase conversation created successfully")

            # Add sample conversation
            conv.add(
                "user", "What are the latest trends in AI technology?"
            )
            conv.add(
                "assistant",
                "Based on current developments, key AI trends include:",
            )
            conv.add(
                "assistant",
                "1. Large Language Models (LLMs) advancing rapidly",
            )
            conv.add(
                "assistant",
                "2. Multimodal AI combining text, image, and video",
            )
            conv.add(
                "assistant",
                "3. AI agents becoming more autonomous and capable",
            )
            conv.add("user", "How do these trends affect businesses?")
            conv.add(
                "assistant",
                "These trends are transforming businesses through automation, enhanced decision-making, and new product capabilities.",
            )

            # Test conversation operations
            print(f"📊 Message count: {len(conv.to_dict())}")
            print(
                f"🔍 Search results for 'AI': {len(conv.search('AI'))}"
            )
            print(
                f"📈 Role distribution: {conv.count_messages_by_role()}"
            )

            # Export conversation
            conv.export_conversation("supabase_ai_conversation.json")
            print(
                "💾 Conversation exported to supabase_ai_conversation.json"
            )

        else:
            print(
                "❌ Skipping Supabase test due to missing configuration"
            )

    except Exception as e:
        print(f"❌ Error in conversation storage test: {e}")

    # Example 3: Agent Creation and Configuration Demo
    print("\n📦 Example 3: Agent Configuration Demo")
    print("-" * 50)

    try:
        if openai_key:
            # Create a simple agent for demonstration
            demo_agent = Agent(
                agent_name="DemoAnalyst",
                agent_description="Demonstration agent for testing",
                system_prompt="You are a helpful AI assistant specializing in analysis and insights.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            )

            print("✅ Demo agent created successfully")
            print(f"Agent: {demo_agent.agent_name}")
            print(f"Description: {demo_agent.agent_description}")

            # Test simple agent run
            simple_task = "Explain the benefits of using persistent conversation storage in AI applications."
            response = demo_agent.run(simple_task)

            print("\n📝 Agent Response:")
            print("-" * 30)
            print(
                response[:500] + "..."
                if len(response) > 500
                else response
            )

        else:
            print(
                "❌ Skipping agent demo due to missing OpenAI API key"
            )

    except Exception as e:
        print(f"❌ Error in agent demo: {e}")

    # Summary and Next Steps
    print("\n" + "=" * 70)
    print("🏁 Demo Summary")
    print("=" * 70)

    print("\n✨ What was demonstrated:")
    print("1. 🏗️  Real Swarms agent creation with specialized roles")
    print("2. 🗄️  Supabase backend integration for persistent storage")
    print("3. 🤝 Multi-agent collaboration and response aggregation")
    print("4. 💾 Conversation export and search capabilities")
    print("5. ⚙️  Proper error handling and graceful fallbacks")

    print("\n🚀 Next Steps to get started:")
    print("1. Set up Supabase project: https://supabase.com")
    print("2. Configure environment variables")
    print("3. Install dependencies: pip install swarms supabase")
    print("4. Customize agents for your specific use cases")
    print("5. Scale to larger agent teams and complex workflows")

    print("\n🔗 Resources:")
    print("- Swarms Documentation: https://docs.swarms.world")
    print(
        "- Supabase Python Docs: https://supabase.com/docs/reference/python/"
    )
    print("- GitHub Repository: https://github.com/kyegomez/swarms")

    print("\n⚙️  Final Configuration Status:")
    print(
        f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Not set'}"
    )
    print(
        f"   SUPABASE_ANON_KEY: {'✅ Set' if supabase_key else '❌ Not set'}"
    )
    print(
        f"   OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}"
    )

    if supabase_url and supabase_key and openai_key:
        print("\n🎉 All systems ready! You can run the full demo.")
    else:
        print(
            "\n⚠️  Set missing environment variables to run the full demo."
        )
