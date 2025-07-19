"""
Enhanced Hierarchical Swarm Example

This example demonstrates the improved capabilities of the EnhancedHierarchicalSwarm including:
- Advanced communication protocols
- Dynamic role assignment
- Intelligent task scheduling
- Performance monitoring
- Parallel execution
"""

from swarms import Agent
from swarms.structs.enhanced_hierarchical_swarm import EnhancedHierarchicalSwarm
import time


def create_research_team():
    """Create a research team with specialized agents"""
    
    # Create specialized research agents
    data_analyst = Agent(
        agent_name="Data-Analyst",
        agent_description="Expert in data analysis, statistical modeling, and data visualization",
        system_prompt="""You are a senior data analyst with expertise in:
        - Statistical analysis and modeling
        - Data visualization and reporting
        - Pattern recognition and insights
        - Database querying and data manipulation
        - Machine learning and predictive analytics
        
        Your role is to analyze data, identify patterns, and provide actionable insights.
        You communicate findings clearly with supporting evidence and visualizations.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.3,
    )
    
    market_researcher = Agent(
        agent_name="Market-Researcher",
        agent_description="Specialist in market research, competitive analysis, and trend identification",
        system_prompt="""You are a senior market researcher with expertise in:
        - Market analysis and competitive intelligence
        - Consumer behavior research
        - Trend identification and forecasting
        - Industry analysis and benchmarking
        - Survey design and data collection
        
        Your role is to research markets, analyze competition, and identify opportunities.
        You provide comprehensive market insights with actionable recommendations.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.3,
    )
    
    technical_writer = Agent(
        agent_name="Technical-Writer",
        agent_description="Expert in technical documentation, report writing, and content creation",
        system_prompt="""You are a senior technical writer with expertise in:
        - Technical documentation and reporting
        - Content creation and editing
        - Information architecture and organization
        - Clear communication of complex topics
        - Research synthesis and summarization
        
        Your role is to create clear, comprehensive documentation and reports.
        You transform complex information into accessible, well-structured content.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.4,
    )
    
    return [data_analyst, market_researcher, technical_writer]


def create_development_team():
    """Create a development team with specialized agents"""
    
    # Create specialized development agents
    backend_developer = Agent(
        agent_name="Backend-Developer",
        agent_description="Expert in backend development, API design, and system architecture",
        system_prompt="""You are a senior backend developer with expertise in:
        - Server-side programming and API development
        - Database design and optimization
        - System architecture and scalability
        - Security implementation and best practices
        - Performance optimization and monitoring
        
        Your role is to design and implement robust backend systems.
        You ensure scalability, security, and performance in all solutions.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.3,
    )
    
    frontend_developer = Agent(
        agent_name="Frontend-Developer",
        agent_description="Expert in frontend development, UI/UX design, and user experience",
        system_prompt="""You are a senior frontend developer with expertise in:
        - Modern JavaScript frameworks and libraries
        - User interface design and implementation
        - User experience optimization
        - Responsive design and accessibility
        - Performance optimization and testing
        
        Your role is to create intuitive, responsive user interfaces.
        You ensure excellent user experience across all platforms.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.3,
    )
    
    devops_engineer = Agent(
        agent_name="DevOps-Engineer",
        agent_description="Expert in DevOps practices, CI/CD, and infrastructure management",
        system_prompt="""You are a senior DevOps engineer with expertise in:
        - Continuous integration and deployment
        - Infrastructure as code and automation
        - Container orchestration and management
        - Monitoring and observability
        - Security and compliance automation
        
        Your role is to streamline development and deployment processes.
        You ensure reliable, scalable, and secure infrastructure.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.3,
    )
    
    return [backend_developer, frontend_developer, devops_engineer]


def run_research_example():
    """Run a comprehensive research example"""
    
    print("🔬 Enhanced Hierarchical Swarm - Research Team Example")
    print("=" * 60)
    
    # Create research team
    research_agents = create_research_team()
    
    # Create enhanced hierarchical swarm
    research_swarm = EnhancedHierarchicalSwarm(
        name="Advanced-Research-Swarm",
        description="Enhanced hierarchical swarm for comprehensive research analysis",
        agents=research_agents,
        max_loops=2,
        verbose=True,
        enable_parallel_execution=True,
        max_concurrent_tasks=5,
        auto_optimize=True
    )
    
    # Define research task
    research_task = """
    Conduct a comprehensive analysis of the electric vehicle (EV) market including:
    1. Market size, growth trends, and future projections
    2. Key players, competitive landscape, and market share analysis
    3. Consumer adoption patterns and barriers
    4. Technological developments and innovations
    5. Regulatory environment and policy impacts
    6. Investment opportunities and risks
    
    Provide detailed findings with data-driven insights and strategic recommendations.
    """
    
    print("🚀 Starting research analysis...")
    start_time = time.time()
    
    # Execute research
    result = research_swarm.run(task=research_task)
    
    execution_time = time.time() - start_time
    print(f"✅ Research completed in {execution_time:.2f} seconds")
    
    # Get performance metrics
    metrics = research_swarm.get_performance_metrics()
    print("\n📊 Performance Metrics:")
    print(f"- Total tasks: {metrics['execution_metrics']['total_tasks']}")
    print(f"- Completed tasks: {metrics['execution_metrics']['completed_tasks']}")
    print(f"- Success rate: {metrics['execution_metrics']['completed_tasks'] / max(1, metrics['execution_metrics']['total_tasks']) * 100:.1f}%")
    print(f"- Average execution time: {metrics['execution_metrics']['avg_execution_time']:.2f}s")
    
    # Display agent performance
    print("\n🤖 Agent Performance:")
    for agent_id, perf in metrics['agent_performance'].items():
        print(f"- {agent_id}:")
        print(f"  Role: {perf['role']}")
        print(f"  Capabilities: {list(perf['capabilities'].keys())}")
        for cap, data in perf['capabilities'].items():
            print(f"    {cap}: skill={data['skill_level']:.2f}, success={data['success_rate']:.2f}")
    
    # Optimize performance
    research_swarm.optimize_performance()
    
    # Shutdown
    research_swarm.shutdown()
    
    return result


def run_development_example():
    """Run a comprehensive development example"""
    
    print("\n💻 Enhanced Hierarchical Swarm - Development Team Example")
    print("=" * 60)
    
    # Create development team
    dev_agents = create_development_team()
    
    # Create enhanced hierarchical swarm
    dev_swarm = EnhancedHierarchicalSwarm(
        name="Advanced-Development-Swarm",
        description="Enhanced hierarchical swarm for software development",
        agents=dev_agents,
        max_loops=3,
        verbose=True,
        enable_parallel_execution=True,
        max_concurrent_tasks=6,
        auto_optimize=True
    )
    
    # Define development task
    dev_task = """
    Design and implement a comprehensive task management system with:
    1. User authentication and authorization
    2. Task creation, assignment, and tracking
    3. Real-time collaboration features
    4. Dashboard with analytics and reporting
    5. Mobile-responsive design
    6. API for third-party integrations
    7. Automated testing and deployment pipeline
    
    Provide detailed technical specifications, implementation plan, and deployment strategy.
    """
    
    print("🚀 Starting development project...")
    start_time = time.time()
    
    # Execute development
    result = dev_swarm.run(task=dev_task)
    
    execution_time = time.time() - start_time
    print(f"✅ Development completed in {execution_time:.2f} seconds")
    
    # Get performance metrics
    metrics = dev_swarm.get_performance_metrics()
    print("\n📊 Performance Metrics:")
    print(f"- Total tasks: {metrics['execution_metrics']['total_tasks']}")
    print(f"- Completed tasks: {metrics['execution_metrics']['completed_tasks']}")
    print(f"- Success rate: {metrics['execution_metrics']['completed_tasks'] / max(1, metrics['execution_metrics']['total_tasks']) * 100:.1f}%")
    print(f"- Average execution time: {metrics['execution_metrics']['avg_execution_time']:.2f}s")
    
    # Display communication statistics
    comm_stats = metrics['communication_stats']
    print("\n📡 Communication Statistics:")
    print(f"- Total channels: {comm_stats['total_channels']}")
    print(f"- Active conversations: {comm_stats['active_conversations']}")
    print(f"- Total agents: {comm_stats['total_agents']}")
    print(f"- Message history size: {comm_stats['message_history_size']}")
    print(f"- Escalation count: {comm_stats['escalation_count']}")
    
    # Optimize performance
    dev_swarm.optimize_performance()
    
    # Shutdown
    dev_swarm.shutdown()
    
    return result


def run_comparative_analysis():
    """Run comparative analysis between different swarm configurations"""
    
    print("\n📈 Comparative Analysis - Standard vs Enhanced Swarm")
    print("=" * 60)
    
    # Create test agents
    test_agents = create_research_team()[:2]  # Use first 2 agents
    
    # Test task
    test_task = "Analyze the current state of renewable energy adoption and provide key insights."
    
    # Test 1: Enhanced swarm with parallel execution
    print("🔄 Test 1: Enhanced Swarm (Parallel)")
    enhanced_parallel = EnhancedHierarchicalSwarm(
        name="Enhanced-Parallel-Swarm",
        agents=test_agents,
        verbose=False,
        enable_parallel_execution=True,
        max_concurrent_tasks=5
    )
    
    start_time = time.time()
    result1 = enhanced_parallel.run(task=test_task)
    time1 = time.time() - start_time
    metrics1 = enhanced_parallel.get_performance_metrics()
    enhanced_parallel.shutdown()
    
    # Test 2: Enhanced swarm with sequential execution
    print("🔄 Test 2: Enhanced Swarm (Sequential)")
    enhanced_sequential = EnhancedHierarchicalSwarm(
        name="Enhanced-Sequential-Swarm",
        agents=test_agents,
        verbose=False,
        enable_parallel_execution=False,
        max_concurrent_tasks=1
    )
    
    start_time = time.time()
    result2 = enhanced_sequential.run(task=test_task)
    time2 = time.time() - start_time
    metrics2 = enhanced_sequential.get_performance_metrics()
    enhanced_sequential.shutdown()
    
    # Compare results
    print("\n📊 Comparison Results:")
    print(f"Parallel Execution: {time1:.2f}s | Sequential Execution: {time2:.2f}s")
    print(f"Performance Improvement: {((time2 - time1) / time2 * 100):.1f}%")
    
    print(f"\nParallel Tasks: {metrics1['execution_metrics']['total_tasks']} | Sequential Tasks: {metrics2['execution_metrics']['total_tasks']}")
    print(f"Parallel Success Rate: {metrics1['execution_metrics']['completed_tasks'] / max(1, metrics1['execution_metrics']['total_tasks']) * 100:.1f}%")
    print(f"Sequential Success Rate: {metrics2['execution_metrics']['completed_tasks'] / max(1, metrics2['execution_metrics']['total_tasks']) * 100:.1f}%")


def main():
    """Main function to run all examples"""
    
    print("🚀 Enhanced Hierarchical Swarm - Comprehensive Examples")
    print("=" * 80)
    
    try:
        # Run research example
        research_result = run_research_example()
        
        # Run development example
        dev_result = run_development_example()
        
        # Run comparative analysis
        run_comparative_analysis()
        
        print("\n🎉 All examples completed successfully!")
        print("=" * 80)
        
        # Summary
        print("\n📋 Summary of Enhanced Capabilities:")
        print("✅ Multi-directional communication between agents")
        print("✅ Dynamic role assignment based on performance")
        print("✅ Intelligent task scheduling and coordination")
        print("✅ Parallel execution for improved performance")
        print("✅ Real-time performance monitoring and optimization")
        print("✅ Advanced error handling and recovery")
        print("✅ Comprehensive metrics and analytics")
        print("✅ Scalable architecture for large teams")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()