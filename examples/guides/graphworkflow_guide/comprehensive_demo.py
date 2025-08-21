#!/usr/bin/env python3
"""
Comprehensive GraphWorkflow Demo Script
=======================================

This script demonstrates all key features of Swarms' GraphWorkflow system,
including parallel processing patterns, performance optimization, and real-world use cases.

Usage:
    python comprehensive_demo.py [--demo healthcare|finance|enterprise|all]

Requirements:
    uv pip install swarms
    uv pip install graphviz  # Optional for visualization
"""

import argparse
import time

from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def create_basic_workflow_demo():
    """Demonstrate basic GraphWorkflow functionality."""

    print("\n" + "=" * 60)
    print("🚀 BASIC GRAPHWORKFLOW DEMONSTRATION")
    print("=" * 60)

    # Create simple agents
    data_collector = Agent(
        agent_name="DataCollector",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a data collection specialist. Gather and organize relevant information for analysis.",
        verbose=False,
    )

    data_analyzer = Agent(
        agent_name="DataAnalyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a data analysis expert. Analyze the collected data and extract key insights.",
        verbose=False,
    )

    report_generator = Agent(
        agent_name="ReportGenerator",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a report generation specialist. Create comprehensive reports from analysis results.",
        verbose=False,
    )

    # Create workflow
    workflow = GraphWorkflow(
        name="BasicWorkflowDemo",
        description="Demonstrates basic GraphWorkflow functionality",
        verbose=True,
        auto_compile=True,
    )

    # Add nodes
    for agent in [data_collector, data_analyzer, report_generator]:
        workflow.add_node(agent)

    # Add edges (sequential flow)
    workflow.add_edge("DataCollector", "DataAnalyzer")
    workflow.add_edge("DataAnalyzer", "ReportGenerator")

    # Set entry and exit points
    workflow.set_entry_points(["DataCollector"])
    workflow.set_end_points(["ReportGenerator"])

    print(
        f"✅ Created workflow with {len(workflow.nodes)} nodes and {len(workflow.edges)} edges"
    )

    # Demonstrate compilation
    compilation_status = workflow.get_compilation_status()
    print(f"📊 Compilation Status: {compilation_status}")

    # Demonstrate simple visualization
    try:
        workflow.visualize_simple()
    except Exception as e:
        print(f"⚠️ Visualization not available: {e}")

    # Run workflow
    task = "Analyze the current state of artificial intelligence in healthcare, focusing on recent developments and future opportunities."

    print(f"\n🔄 Executing workflow with task: {task[:100]}...")
    start_time = time.time()

    results = workflow.run(task=task)

    execution_time = time.time() - start_time
    print(f"⏱️ Execution completed in {execution_time:.2f} seconds")

    # Display results
    print("\n📋 Results Summary:")
    for agent_name, result in results.items():
        print(f"\n🤖 {agent_name}:")
        print(
            f"   {result[:200]}{'...' if len(result) > 200 else ''}"
        )

    return workflow, results


def create_parallel_processing_demo():
    """Demonstrate advanced parallel processing patterns."""

    print("\n" + "=" * 60)
    print("⚡ PARALLEL PROCESSING DEMONSTRATION")
    print("=" * 60)

    # Create data sources
    web_scraper = Agent(
        agent_name="WebScraper",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in web data scraping and online research.",
        verbose=False,
    )

    api_collector = Agent(
        agent_name="APICollector",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in API data collection and integration.",
        verbose=False,
    )

    database_extractor = Agent(
        agent_name="DatabaseExtractor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in database queries and data extraction.",
        verbose=False,
    )

    # Create parallel processors
    text_processor = Agent(
        agent_name="TextProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in natural language processing and text analysis.",
        verbose=False,
    )

    numeric_processor = Agent(
        agent_name="NumericProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in numerical analysis and statistical processing.",
        verbose=False,
    )

    # Create analyzers
    sentiment_analyzer = Agent(
        agent_name="SentimentAnalyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in sentiment analysis and emotional intelligence.",
        verbose=False,
    )

    trend_analyzer = Agent(
        agent_name="TrendAnalyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in trend analysis and pattern recognition.",
        verbose=False,
    )

    # Create synthesizer
    data_synthesizer = Agent(
        agent_name="DataSynthesizer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You specialize in data synthesis and comprehensive analysis integration.",
        verbose=False,
    )

    # Create workflow
    workflow = GraphWorkflow(
        name="ParallelProcessingDemo",
        description="Demonstrates advanced parallel processing patterns including fan-out, fan-in, and parallel chains",
        verbose=True,
        auto_compile=True,
    )

    # Add all agents
    agents = [
        web_scraper,
        api_collector,
        database_extractor,
        text_processor,
        numeric_processor,
        sentiment_analyzer,
        trend_analyzer,
        data_synthesizer,
    ]

    for agent in agents:
        workflow.add_node(agent)

    # Demonstrate different parallel patterns
    print("🔀 Setting up parallel processing patterns...")

    # Pattern 1: Fan-out from sources to processors
    print("   📤 Fan-out: Data sources → Processors")
    workflow.add_edges_from_source(
        "WebScraper", ["TextProcessor", "SentimentAnalyzer"]
    )
    workflow.add_edges_from_source(
        "APICollector", ["NumericProcessor", "TrendAnalyzer"]
    )
    workflow.add_edges_from_source(
        "DatabaseExtractor", ["TextProcessor", "NumericProcessor"]
    )

    # Pattern 2: Parallel chain from processors to analyzers
    print("   🔗 Parallel chain: Processors → Analyzers")
    workflow.add_parallel_chain(
        ["TextProcessor", "NumericProcessor"],
        ["SentimentAnalyzer", "TrendAnalyzer"],
    )

    # Pattern 3: Fan-in to synthesizer
    print("   📥 Fan-in: All analyzers → Synthesizer")
    workflow.add_edges_to_target(
        ["SentimentAnalyzer", "TrendAnalyzer"], "DataSynthesizer"
    )

    # Set entry and exit points
    workflow.set_entry_points(
        ["WebScraper", "APICollector", "DatabaseExtractor"]
    )
    workflow.set_end_points(["DataSynthesizer"])

    print(
        f"✅ Created parallel workflow with {len(workflow.nodes)} nodes and {len(workflow.edges)} edges"
    )

    # Analyze parallel patterns
    compilation_status = workflow.get_compilation_status()
    print(f"📊 Compilation Status: {compilation_status}")
    print(
        f"🔧 Execution layers: {len(compilation_status.get('layers', []))}"
    )
    print(
        f"⚡ Max parallel workers: {compilation_status.get('max_workers', 'N/A')}"
    )

    # Run parallel workflow
    task = "Research and analyze the impact of quantum computing on cybersecurity, examining technical developments, market trends, and security implications."

    print("\n🔄 Executing parallel workflow...")
    start_time = time.time()

    results = workflow.run(task=task)

    execution_time = time.time() - start_time
    print(
        f"⏱️ Parallel execution completed in {execution_time:.2f} seconds"
    )
    print(
        f"🚀 Throughput: {len(results)/execution_time:.1f} agents/second"
    )

    # Display results
    print("\n📋 Parallel Processing Results:")
    for agent_name, result in results.items():
        print(f"\n🤖 {agent_name}:")
        print(
            f"   {result[:150]}{'...' if len(result) > 150 else ''}"
        )

    return workflow, results


def create_healthcare_workflow_demo():
    """Demonstrate healthcare-focused workflow."""

    print("\n" + "=" * 60)
    print("🏥 HEALTHCARE WORKFLOW DEMONSTRATION")
    print("=" * 60)

    # Create clinical specialists
    primary_care_physician = Agent(
        agent_name="PrimaryCarePhysician",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a board-certified primary care physician. Provide:
        1. Initial patient assessment and history taking
        2. Differential diagnosis development
        3. Treatment plan coordination
        4. Preventive care recommendations
        
        Focus on comprehensive, evidence-based primary care.""",
        verbose=False,
    )

    cardiologist = Agent(
        agent_name="Cardiologist",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a board-certified cardiologist. Provide:
        1. Cardiovascular risk assessment
        2. Cardiac diagnostic interpretation
        3. Treatment recommendations for heart conditions
        4. Cardiovascular prevention strategies
        
        Apply evidence-based cardiology guidelines.""",
        verbose=False,
    )

    pharmacist = Agent(
        agent_name="ClinicalPharmacist",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a clinical pharmacist specialist. Provide:
        1. Medication review and optimization
        2. Drug interaction analysis
        3. Dosing recommendations
        4. Patient counseling guidance
        
        Ensure medication safety and efficacy.""",
        verbose=False,
    )

    case_manager = Agent(
        agent_name="CaseManager",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a clinical case manager. Coordinate:
        1. Care plan integration and implementation
        2. Resource allocation and scheduling
        3. Patient education and follow-up
        4. Quality metrics and outcomes tracking
        
        Ensure coordinated, patient-centered care.""",
        verbose=False,
    )

    # Create workflow
    workflow = GraphWorkflow(
        name="HealthcareWorkflowDemo",
        description="Clinical decision support workflow with multi-disciplinary team collaboration",
        verbose=True,
        auto_compile=True,
    )

    # Add agents
    agents = [
        primary_care_physician,
        cardiologist,
        pharmacist,
        case_manager,
    ]
    for agent in agents:
        workflow.add_node(agent)

    # Create clinical workflow
    workflow.add_edge("PrimaryCarePhysician", "Cardiologist")
    workflow.add_edge("PrimaryCarePhysician", "ClinicalPharmacist")
    workflow.add_edges_to_target(
        ["Cardiologist", "ClinicalPharmacist"], "CaseManager"
    )

    workflow.set_entry_points(["PrimaryCarePhysician"])
    workflow.set_end_points(["CaseManager"])

    print(
        f"✅ Created healthcare workflow with {len(workflow.nodes)} specialists"
    )

    # Clinical case
    clinical_case = """
    Patient: 58-year-old male executive
    Chief Complaint: Chest pain and shortness of breath during exercise
    History: Hypertension, family history of coronary artery disease, sedentary lifestyle
    Current Medications: Lisinopril 10mg daily
    Vital Signs: BP 145/92, HR 88, BMI 29.5
    Recent Tests: ECG shows non-specific changes, cholesterol 245 mg/dL
    
    Please provide comprehensive clinical assessment and care coordination.
    """

    print("\n🔄 Processing clinical case...")
    start_time = time.time()

    results = workflow.run(task=clinical_case)

    execution_time = time.time() - start_time
    print(
        f"⏱️ Clinical assessment completed in {execution_time:.2f} seconds"
    )

    # Display clinical results
    print("\n🏥 Clinical Team Assessment:")
    for agent_name, result in results.items():
        print(f"\n👨‍⚕️ {agent_name}:")
        print(
            f"   📋 {result[:200]}{'...' if len(result) > 200 else ''}"
        )

    return workflow, results


def create_finance_workflow_demo():
    """Demonstrate finance-focused workflow."""

    print("\n" + "=" * 60)
    print("💰 FINANCE WORKFLOW DEMONSTRATION")
    print("=" * 60)

    # Create financial analysts
    market_analyst = Agent(
        agent_name="MarketAnalyst",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a senior market analyst. Provide:
        1. Market condition assessment and trends
        2. Sector rotation and thematic analysis
        3. Economic indicator interpretation
        4. Market timing and positioning recommendations
        
        Apply rigorous market analysis frameworks.""",
        verbose=False,
    )

    equity_researcher = Agent(
        agent_name="EquityResearcher",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are an equity research analyst. Provide:
        1. Company fundamental analysis
        2. Financial modeling and valuation
        3. Competitive positioning assessment
        4. Investment thesis development
        
        Use comprehensive equity research methodologies.""",
        verbose=False,
    )

    risk_manager = Agent(
        agent_name="RiskManager",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a risk management specialist. Provide:
        1. Portfolio risk assessment and metrics
        2. Stress testing and scenario analysis
        3. Risk mitigation strategies
        4. Regulatory compliance guidance
        
        Apply quantitative risk management principles.""",
        verbose=False,
    )

    portfolio_manager = Agent(
        agent_name="PortfolioManager",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a senior portfolio manager. Provide:
        1. Investment decision synthesis
        2. Portfolio construction and allocation
        3. Performance attribution analysis
        4. Client communication and reporting
        
        Integrate all analysis into actionable investment decisions.""",
        verbose=False,
    )

    # Create workflow
    workflow = GraphWorkflow(
        name="FinanceWorkflowDemo",
        description="Investment decision workflow with multi-disciplinary financial analysis",
        verbose=True,
        auto_compile=True,
    )

    # Add agents
    agents = [
        market_analyst,
        equity_researcher,
        risk_manager,
        portfolio_manager,
    ]
    for agent in agents:
        workflow.add_node(agent)

    # Create financial workflow (parallel analysis feeding portfolio decisions)
    workflow.add_edges_from_source(
        "MarketAnalyst", ["EquityResearcher", "RiskManager"]
    )
    workflow.add_edges_to_target(
        ["EquityResearcher", "RiskManager"], "PortfolioManager"
    )

    workflow.set_entry_points(["MarketAnalyst"])
    workflow.set_end_points(["PortfolioManager"])

    print(
        f"✅ Created finance workflow with {len(workflow.nodes)} analysts"
    )

    # Investment analysis task
    investment_scenario = """
    Investment Analysis Request: Technology Sector Allocation
    
    Market Context:
    - Interest rates: 5.25% federal funds rate
    - Inflation: 3.2% CPI year-over-year
    - Technology sector: -8% YTD performance
    - AI theme: High investor interest and valuation concerns
    
    Portfolio Context:
    - Current tech allocation: 15% (target 20-25%)
    - Risk budget: 12% tracking error limit
    - Investment horizon: 3-5 years
    - Client risk tolerance: Moderate-aggressive
    
    Please provide comprehensive investment analysis and recommendations.
    """

    print("\n🔄 Analyzing investment scenario...")
    start_time = time.time()

    results = workflow.run(task=investment_scenario)

    execution_time = time.time() - start_time
    print(
        f"⏱️ Investment analysis completed in {execution_time:.2f} seconds"
    )

    # Display financial results
    print("\n💼 Investment Team Analysis:")
    for agent_name, result in results.items():
        print(f"\n📈 {agent_name}:")
        print(
            f"   💡 {result[:200]}{'...' if len(result) > 200 else ''}"
        )

    return workflow, results


def demonstrate_serialization_features():
    """Demonstrate workflow serialization and persistence."""

    print("\n" + "=" * 60)
    print("💾 SERIALIZATION & PERSISTENCE DEMONSTRATION")
    print("=" * 60)

    # Create a simple workflow for serialization demo
    agent1 = Agent(
        agent_name="SerializationTestAgent1",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are agent 1 for serialization testing.",
        verbose=False,
    )

    agent2 = Agent(
        agent_name="SerializationTestAgent2",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are agent 2 for serialization testing.",
        verbose=False,
    )

    # Create workflow
    workflow = GraphWorkflow(
        name="SerializationTestWorkflow",
        description="Workflow for testing serialization capabilities",
        verbose=True,
        auto_compile=True,
    )

    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_edge(
        "SerializationTestAgent1", "SerializationTestAgent2"
    )

    print("✅ Created test workflow for serialization")

    # Test JSON serialization
    print("\n📄 Testing JSON serialization...")
    try:
        json_data = workflow.to_json(
            include_conversation=True, include_runtime_state=True
        )
        print(
            f"✅ JSON serialization successful ({len(json_data)} characters)"
        )

        # Test deserialization
        print("\n📥 Testing JSON deserialization...")
        restored_workflow = GraphWorkflow.from_json(
            json_data, restore_runtime_state=True
        )
        print("✅ JSON deserialization successful")
        print(
            f"   Restored {len(restored_workflow.nodes)} nodes, {len(restored_workflow.edges)} edges"
        )

    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")

    # Test file persistence
    print("\n💾 Testing file persistence...")
    try:
        filepath = workflow.save_to_file(
            "test_workflow.json",
            include_conversation=True,
            include_runtime_state=True,
            overwrite=True,
        )
        print(f"✅ File save successful: {filepath}")

        # Test file loading
        loaded_workflow = GraphWorkflow.load_from_file(
            filepath, restore_runtime_state=True
        )
        print("✅ File load successful")
        print(
            f"   Loaded {len(loaded_workflow.nodes)} nodes, {len(loaded_workflow.edges)} edges"
        )

        # Clean up
        import os

        os.remove(filepath)
        print("🧹 Cleaned up test file")

    except Exception as e:
        print(f"❌ File persistence failed: {e}")

    # Test workflow validation
    print("\n🔍 Testing workflow validation...")
    try:
        validation_result = workflow.validate(auto_fix=True)
        print("✅ Validation completed")
        print(f"   Valid: {validation_result['is_valid']}")
        print(f"   Warnings: {len(validation_result['warnings'])}")
        print(f"   Errors: {len(validation_result['errors'])}")
        if validation_result["fixed"]:
            print(f"   Auto-fixed: {validation_result['fixed']}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")


def demonstrate_visualization_features():
    """Demonstrate workflow visualization capabilities."""

    print("\n" + "=" * 60)
    print("🎨 VISUALIZATION DEMONSTRATION")
    print("=" * 60)

    # Create a workflow with interesting patterns for visualization
    workflow = GraphWorkflow(
        name="VisualizationDemo",
        description="Workflow designed to showcase visualization capabilities",
        verbose=True,
        auto_compile=True,
    )

    # Create agents with different roles
    agents = []
    for i, role in enumerate(
        ["DataSource", "Processor", "Analyzer", "Reporter"], 1
    ):
        for j in range(2):
            agent = Agent(
                agent_name=f"{role}{j+1}",
                model_name="gpt-4o-mini",
                max_loops=1,
                system_prompt=f"You are {role} #{j+1}",
                verbose=False,
            )
            agents.append(agent)
            workflow.add_node(agent)

    # Create interesting edge patterns
    # Fan-out from data sources
    workflow.add_edges_from_source(
        "DataSource1", ["Processor1", "Processor2"]
    )
    workflow.add_edges_from_source(
        "DataSource2", ["Processor1", "Processor2"]
    )

    # Parallel processing
    workflow.add_parallel_chain(
        ["Processor1", "Processor2"], ["Analyzer1", "Analyzer2"]
    )

    # Fan-in to reporters
    workflow.add_edges_to_target(
        ["Analyzer1", "Analyzer2"], "Reporter1"
    )
    workflow.add_edge("Analyzer1", "Reporter2")

    print(
        f"✅ Created visualization demo workflow with {len(workflow.nodes)} nodes"
    )

    # Test text visualization (always available)
    print("\n📝 Testing text visualization...")
    try:
        workflow.visualize_simple()
        print("✅ Text visualization successful")
    except Exception as e:
        print(f"❌ Text visualization failed: {e}")

    # Test Graphviz visualization (if available)
    print("\n🎨 Testing Graphviz visualization...")
    try:
        viz_path = workflow.visualize(
            format="png", view=False, show_summary=True
        )
        print(f"✅ Graphviz visualization successful: {viz_path}")
    except ImportError:
        print(
            "⚠️ Graphviz not available - skipping advanced visualization"
        )
    except Exception as e:
        print(f"❌ Graphviz visualization failed: {e}")

    # Export workflow summary
    print("\n📊 Generating workflow summary...")
    try:
        summary = workflow.export_summary()
        print("✅ Workflow summary generated")
        print(f"   Structure: {summary['structure']}")
        print(f"   Configuration: {summary['configuration']}")
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")


def run_performance_benchmarks():
    """Run performance benchmarks comparing different execution strategies."""

    print("\n" + "=" * 60)
    print("🏃‍♂️ PERFORMANCE BENCHMARKING")
    print("=" * 60)

    # Create workflows of different sizes
    sizes = [5, 10, 15]
    results = {}

    for size in sizes:
        print(f"\n📊 Benchmarking workflow with {size} agents...")

        # Create workflow
        workflow = GraphWorkflow(
            name=f"BenchmarkWorkflow{size}",
            description=f"Benchmark workflow with {size} agents",
            verbose=False,  # Reduce logging for benchmarks
            auto_compile=True,
        )

        # Create agents
        agents = []
        for i in range(size):
            agent = Agent(
                agent_name=f"BenchmarkAgent{i+1}",
                model_name="gpt-4o-mini",
                max_loops=1,
                system_prompt=f"You are benchmark agent {i+1}. Provide a brief analysis.",
                verbose=False,
            )
            agents.append(agent)
            workflow.add_node(agent)

        # Create simple sequential workflow
        for i in range(size - 1):
            workflow.add_edge(
                f"BenchmarkAgent{i+1}", f"BenchmarkAgent{i+2}"
            )

        # Benchmark compilation
        compile_start = time.time()
        workflow.compile()
        compile_time = time.time() - compile_start

        # Benchmark execution
        task = (
            "Provide a brief analysis of current market conditions."
        )

        exec_start = time.time()
        exec_results = workflow.run(task=task)
        exec_time = time.time() - exec_start

        # Store results
        results[size] = {
            "compile_time": compile_time,
            "execution_time": exec_time,
            "agents_executed": len(exec_results),
            "throughput": (
                len(exec_results) / exec_time if exec_time > 0 else 0
            ),
        }

        print(f"   ⏱️ Compilation: {compile_time:.3f}s")
        print(f"   ⏱️ Execution: {exec_time:.3f}s")
        print(
            f"   🚀 Throughput: {results[size]['throughput']:.1f} agents/second"
        )

    # Display benchmark summary
    print("\n📈 PERFORMANCE BENCHMARK SUMMARY")
    print("-" * 50)
    print(
        f"{'Size':<6} {'Compile(s)':<12} {'Execute(s)':<12} {'Throughput':<12}"
    )
    print("-" * 50)

    for size, metrics in results.items():
        print(
            f"{size:<6} {metrics['compile_time']:<12.3f} {metrics['execution_time']:<12.3f} {metrics['throughput']:<12.1f}"
        )

    return results


def main():
    """Main demonstration function."""

    parser = argparse.ArgumentParser(
        description="GraphWorkflow Comprehensive Demo"
    )
    parser.add_argument(
        "--demo",
        choices=[
            "basic",
            "parallel",
            "healthcare",
            "finance",
            "serialization",
            "visualization",
            "performance",
            "all",
        ],
        default="all",
        help="Which demonstration to run",
    )

    args = parser.parse_args()

    print("🌟 SWARMS GRAPHWORKFLOW COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print(
        "The LangGraph Killer: Advanced Multi-Agent Workflow Orchestration"
    )
    print("=" * 70)

    demos = {
        "basic": create_basic_workflow_demo,
        "parallel": create_parallel_processing_demo,
        "healthcare": create_healthcare_workflow_demo,
        "finance": create_finance_workflow_demo,
        "serialization": demonstrate_serialization_features,
        "visualization": demonstrate_visualization_features,
        "performance": run_performance_benchmarks,
    }

    if args.demo == "all":
        # Run all demonstrations
        for demo_name, demo_func in demos.items():
            try:
                print(f"\n🎯 Running {demo_name} demonstration...")
                demo_func()
            except Exception as e:
                print(f"❌ {demo_name} demonstration failed: {e}")
    else:
        # Run specific demonstration
        if args.demo in demos:
            try:
                demos[args.demo]()
            except Exception as e:
                print(f"❌ Demonstration failed: {e}")
        else:
            print(f"❌ Unknown demonstration: {args.demo}")

    print("\n" + "=" * 70)
    print("🎉 DEMONSTRATION COMPLETED")
    print("=" * 70)
    print(
        "GraphWorkflow provides enterprise-grade multi-agent orchestration"
    )
    print("with superior performance, reliability, and ease of use.")
    print("\nNext steps:")
    print("1. Try the healthcare or finance examples in your domain")
    print("2. Experiment with parallel processing patterns")
    print("3. Deploy to production with monitoring and optimization")
    print(
        "4. Explore advanced features like caching and serialization"
    )


if __name__ == "__main__":
    main()
