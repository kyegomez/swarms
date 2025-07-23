"""
Board of Directors Example

This example demonstrates how to use the Board of Directors swarm feature
in the Swarms Framework. It shows how to create a board, configure it,
and use it to orchestrate tasks across multiple agents.

The example includes:
1. Basic Board of Directors setup and usage
2. Custom board member configuration
3. Task execution and feedback
4. Configuration management
"""

import os
import sys
from typing import List, Optional

# Add the parent directory to the path to import swarms
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole,
)
from swarms.structs.agent import Agent
from swarms.config.board_config import (
    enable_board_feature,
    disable_board_feature,
    is_board_feature_enabled,
    create_default_config_file,
    set_board_size,
    set_decision_threshold,
    set_board_model,
    enable_verbose_logging,
    disable_verbose_logging,
)


def enable_board_directors_feature() -> None:
    """
    Enable the Board of Directors feature.
    
    This function demonstrates how to enable the Board of Directors feature
    globally and create a default configuration file.
    """
    print("🔧 Enabling Board of Directors feature...")
    
    try:
        # Create a default configuration file
        create_default_config_file("swarms_board_config.yaml")
        
        # Enable the feature
        enable_board_feature("swarms_board_config.yaml")
        
        # Configure some default settings
        set_board_size(3)
        set_decision_threshold(0.6)
        set_board_model("gpt-4o-mini")
        enable_verbose_logging("swarms_board_config.yaml")
        
        print("✅ Board of Directors feature enabled successfully!")
        print("📁 Configuration file created: swarms_board_config.yaml")
        
    except Exception as e:
        print(f"❌ Failed to enable Board of Directors feature: {e}")
        raise


def create_custom_board_members() -> List[BoardMember]:
    """
    Create custom board members with specific roles and expertise.
    
    This function demonstrates how to create a custom board with
    specialized roles and expertise areas.
    
    Returns:
        List[BoardMember]: List of custom board members
    """
    print("👥 Creating custom board members...")
    
    # Create specialized board members
    chairman = Agent(
        agent_name="Executive_Chairman",
        agent_description="Executive Chairman with strategic vision and leadership expertise",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are the Executive Chairman of the Board. Your role is to:
1. Provide strategic leadership and vision
2. Facilitate high-level decision-making
3. Ensure board effectiveness and governance
4. Represent the organization's interests
5. Guide long-term strategic planning

You should be visionary, strategic, and focused on organizational success.""",
    )
    
    cto = Agent(
        agent_name="CTO",
        agent_description="Chief Technology Officer with deep technical expertise",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are the Chief Technology Officer. Your role is to:
1. Provide technical leadership and strategy
2. Evaluate technology solutions and architectures
3. Ensure technical feasibility of proposed solutions
4. Guide technology-related decisions
5. Maintain technical standards and best practices

You should be technically proficient, innovative, and focused on technical excellence.""",
    )
    
    cfo = Agent(
        agent_name="CFO",
        agent_description="Chief Financial Officer with financial and risk management expertise",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are the Chief Financial Officer. Your role is to:
1. Provide financial analysis and insights
2. Evaluate financial implications of decisions
3. Ensure financial sustainability and risk management
4. Guide resource allocation and budgeting
5. Maintain financial controls and compliance

You should be financially astute, risk-aware, and focused on financial health.""",
    )
    
    # Create BoardMember objects with roles and expertise
    board_members = [
        BoardMember(
            agent=chairman,
            role=BoardMemberRole.CHAIRMAN,
            voting_weight=2.0,
            expertise_areas=["strategic_planning", "leadership", "governance", "business_strategy"]
        ),
        BoardMember(
            agent=cto,
            role=BoardMemberRole.EXECUTIVE_DIRECTOR,
            voting_weight=1.5,
            expertise_areas=["technology", "architecture", "innovation", "technical_strategy"]
        ),
        BoardMember(
            agent=cfo,
            role=BoardMemberRole.EXECUTIVE_DIRECTOR,
            voting_weight=1.5,
            expertise_areas=["finance", "risk_management", "budgeting", "financial_analysis"]
        ),
    ]
    
    print(f"✅ Created {len(board_members)} custom board members")
    for member in board_members:
        print(f"   - {member.agent.agent_name} ({member.role.value})")
    
    return board_members


def create_worker_agents() -> List[Agent]:
    """
    Create worker agents for the swarm.
    
    This function creates specialized worker agents that will be
    managed by the Board of Directors.
    
    Returns:
        List[Agent]: List of worker agents
    """
    print("🛠️ Creating worker agents...")
    
    # Create specialized worker agents
    researcher = Agent(
        agent_name="Research_Analyst",
        agent_description="Research analyst specializing in market research and data analysis",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a Research Analyst. Your responsibilities include:
1. Conducting thorough research on assigned topics
2. Analyzing data and market trends
3. Preparing comprehensive research reports
4. Providing data-driven insights and recommendations
5. Maintaining high standards of research quality

You should be analytical, thorough, and evidence-based in your work.""",
    )
    
    developer = Agent(
        agent_name="Software_Developer",
        agent_description="Software developer with expertise in system design and implementation",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a Software Developer. Your responsibilities include:
1. Designing and implementing software solutions
2. Writing clean, maintainable code
3. Conducting code reviews and testing
4. Collaborating with team members
5. Following best practices and coding standards

You should be technically skilled, detail-oriented, and focused on quality.""",
    )
    
    marketer = Agent(
        agent_name="Marketing_Specialist",
        agent_description="Marketing specialist with expertise in digital marketing and brand strategy",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a Marketing Specialist. Your responsibilities include:
1. Developing marketing strategies and campaigns
2. Creating compelling content and messaging
3. Analyzing market trends and customer behavior
4. Managing brand presence and reputation
5. Measuring and optimizing marketing performance

You should be creative, strategic, and customer-focused in your approach.""",
    )
    
    agents = [researcher, developer, marketer]
    
    print(f"✅ Created {len(agents)} worker agents")
    for agent in agents:
        print(f"   - {agent.agent_name}: {agent.agent_description}")
    
    return agents


def run_board_of_directors_example() -> None:
    """
    Run a comprehensive Board of Directors example.
    
    This function demonstrates the complete workflow of using
    the Board of Directors swarm to orchestrate tasks.
    """
    print("\n" + "="*60)
    print("🏛️ BOARD OF DIRECTORS SWARM EXAMPLE")
    print("="*60)
    
    try:
        # Check if Board of Directors feature is enabled
        if not is_board_feature_enabled():
            print("⚠️ Board of Directors feature is not enabled. Enabling now...")
            enable_board_directors_feature()
        
        # Create custom board members
        board_members = create_custom_board_members()
        
        # Create worker agents
        worker_agents = create_worker_agents()
        
        # Create the Board of Directors swarm
        print("\n🏛️ Creating Board of Directors swarm...")
        board_swarm = BoardOfDirectorsSwarm(
            name="Executive_Board_Swarm",
            description="Executive board with specialized roles for strategic decision-making",
            board_members=board_members,
            agents=worker_agents,
            max_loops=2,
            verbose=True,
            decision_threshold=0.6,
            enable_voting=True,
            enable_consensus=True,
        )
        
        print("✅ Board of Directors swarm created successfully!")
        
        # Display board summary
        summary = board_swarm.get_board_summary()
        print(f"\n📊 Board Summary:")
        print(f"   Board Name: {summary['board_name']}")
        print(f"   Total Members: {summary['total_members']}")
        print(f"   Total Agents: {summary['total_agents']}")
        print(f"   Max Loops: {summary['max_loops']}")
        print(f"   Decision Threshold: {summary['decision_threshold']}")
        
        print(f"\n👥 Board Members:")
        for member in summary['members']:
            print(f"   - {member['name']} ({member['role']}) - Weight: {member['voting_weight']}")
            print(f"     Expertise: {', '.join(member['expertise_areas'])}")
        
        # Define a complex task for the board to handle
        task = """
        Develop a comprehensive strategy for launching a new AI-powered product in the market.
        
        The task involves:
        1. Market research and competitive analysis
        2. Technical architecture and development planning
        3. Marketing strategy and go-to-market plan
        4. Financial projections and risk assessment
        
        Please coordinate the efforts of all team members to create a cohesive strategy.
        """
        
        print(f"\n📋 Executing task: {task.strip()[:100]}...")
        
        # Execute the task using the Board of Directors swarm
        result = board_swarm.run(task=task)
        
        print("\n✅ Task completed successfully!")
        print(f"📄 Result type: {type(result)}")
        
        # Display conversation history
        if hasattr(result, 'get') and callable(result.get):
            conversation_history = result.get('conversation_history', [])
            print(f"\n💬 Conversation History ({len(conversation_history)} messages):")
            for i, message in enumerate(conversation_history[-5:], 1):  # Show last 5 messages
                role = message.get('role', 'Unknown')
                content = message.get('content', '')[:100] + "..." if len(message.get('content', '')) > 100 else message.get('content', '')
                print(f"   {i}. {role}: {content}")
        else:
            print(f"\n📝 Result: {str(result)[:200]}...")
        
        print("\n🎉 Board of Directors example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in Board of Directors example: {e}")
        import traceback
        traceback.print_exc()


def run_simple_board_example() -> None:
    """
    Run a simple Board of Directors example with default settings.
    
    This function demonstrates a basic usage of the Board of Directors
    swarm with minimal configuration.
    """
    print("\n" + "="*60)
    print("🏛️ SIMPLE BOARD OF DIRECTORS EXAMPLE")
    print("="*60)
    
    try:
        # Create simple worker agents
        print("🛠️ Creating simple worker agents...")
        
        analyst = Agent(
            agent_name="Data_Analyst",
            agent_description="Data analyst for processing and analyzing information",
            model_name="gpt-4o-mini",
            max_loops=1,
        )
        
        writer = Agent(
            agent_name="Content_Writer",
            agent_description="Content writer for creating reports and documentation",
            model_name="gpt-4o-mini",
            max_loops=1,
        )
        
        agents = [analyst, writer]
        
        # Create Board of Directors swarm with default settings
        print("🏛️ Creating Board of Directors swarm with default settings...")
        board_swarm = BoardOfDirectorsSwarm(
            name="Simple_Board_Swarm",
            agents=agents,
            verbose=True,
        )
        
        print("✅ Simple Board of Directors swarm created!")
        
        # Simple task
        task = "Analyze the current market trends and create a summary report with recommendations."
        
        print(f"\n📋 Executing simple task: {task}")
        
        # Execute the task
        result = board_swarm.run(task=task)
        
        print("\n✅ Simple task completed successfully!")
        print(f"📄 Result type: {type(result)}")
        
        if hasattr(result, 'get') and callable(result.get):
            conversation_history = result.get('conversation_history', [])
            print(f"\n💬 Conversation History ({len(conversation_history)} messages):")
            for i, message in enumerate(conversation_history[-3:], 1):  # Show last 3 messages
                role = message.get('role', 'Unknown')
                content = message.get('content', '')[:80] + "..." if len(message.get('content', '')) > 80 else message.get('content', '')
                print(f"   {i}. {role}: {content}")
        else:
            print(f"\n📝 Result: {str(result)[:150]}...")
        
        print("\n🎉 Simple Board of Directors example completed!")
        
    except Exception as e:
        print(f"❌ Error in simple Board of Directors example: {e}")
        import traceback
        traceback.print_exc()


def check_environment() -> bool:
    """
    Check if the environment is properly set up for the example.
    
    Returns:
        bool: True if environment is ready, False otherwise
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY environment variable not set.")
        print("   The example may not work without a valid API key.")
        print("   Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return False
    
    return True


def main() -> None:
    """
    Main function to run the Board of Directors examples.
    """
    print("🚀 Board of Directors Swarm Examples")
    print("="*50)
    
    # Check environment
    if not check_environment():
        print("\n⚠️ Environment check failed. Please set up your environment properly.")
        return
    
    try:
        # Run simple example first
        run_simple_board_example()
        
        # Run comprehensive example
        run_board_of_directors_example()
        
        print("\n" + "="*60)
        print("🎉 All Board of Directors examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()