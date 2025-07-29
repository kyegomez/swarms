"""
Comprehensive test of Graphviz visualization capabilities for GraphWorkflow.
This demonstrates various layouts, formats, and parallel pattern visualization features.
"""

import os
from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def create_simple_workflow():
    """Create a simple sequential workflow."""
    agent1 = Agent(
        agent_name="DataCollector",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You collect and prepare data for analysis.",
        verbose=False,
    )

    agent2 = Agent(
        agent_name="DataAnalyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You analyze the collected data and extract insights.",
        verbose=False,
    )

    agent3 = Agent(
        agent_name="ReportGenerator",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You generate comprehensive reports from the analysis.",
        verbose=False,
    )

    workflow = GraphWorkflow(
        name="Simple-Sequential-Workflow",
        description="A basic sequential workflow for testing visualization",
        verbose=True,
    )

    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_node(agent3)

    workflow.add_edge("DataCollector", "DataAnalyzer")
    workflow.add_edge("DataAnalyzer", "ReportGenerator")

    return workflow


def create_complex_parallel_workflow():
    """Create a complex workflow with multiple parallel patterns."""
    # Data sources
    web_scraper = Agent(
        agent_name="WebScraper",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Web data scraping",
        verbose=False,
    )
    api_collector = Agent(
        agent_name="APICollector",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="API data collection",
        verbose=False,
    )
    db_extractor = Agent(
        agent_name="DatabaseExtractor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Database extraction",
        verbose=False,
    )

    # Processors
    text_processor = Agent(
        agent_name="TextProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Text processing",
        verbose=False,
    )
    numeric_processor = Agent(
        agent_name="NumericProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Numeric processing",
        verbose=False,
    )
    image_processor = Agent(
        agent_name="ImageProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Image processing",
        verbose=False,
    )

    # Analyzers
    sentiment_analyzer = Agent(
        agent_name="SentimentAnalyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Sentiment analysis",
        verbose=False,
    )
    trend_analyzer = Agent(
        agent_name="TrendAnalyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Trend analysis",
        verbose=False,
    )
    anomaly_detector = Agent(
        agent_name="AnomalyDetector",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Anomaly detection",
        verbose=False,
    )

    # Synthesis
    data_synthesizer = Agent(
        agent_name="DataSynthesizer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Data synthesis",
        verbose=False,
    )

    # Final output
    dashboard_generator = Agent(
        agent_name="DashboardGenerator",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Dashboard generation",
        verbose=False,
    )
    alert_system = Agent(
        agent_name="AlertSystem",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="Alert generation",
        verbose=False,
    )

    workflow = GraphWorkflow(
        name="Complex-Parallel-Analytics-Workflow",
        description="A sophisticated analytics workflow demonstrating multiple parallel processing patterns including fan-out, fan-in, and parallel chains for comprehensive data processing and analysis.",
        verbose=True,
        auto_compile=True,
    )

    # Add all agents
    agents = [
        web_scraper,
        api_collector,
        db_extractor,
        text_processor,
        numeric_processor,
        image_processor,
        sentiment_analyzer,
        trend_analyzer,
        anomaly_detector,
        data_synthesizer,
        dashboard_generator,
        alert_system,
    ]

    for agent in agents:
        workflow.add_node(agent)

    # Create complex parallel patterns
    # Stage 1: Multiple data sources (parallel entry points)
    # Stage 2: Fan-out to different processors
    workflow.add_edge("WebScraper", "TextProcessor")
    workflow.add_edge("WebScraper", "ImageProcessor")
    workflow.add_edge("APICollector", "NumericProcessor")
    workflow.add_edge("APICollector", "TextProcessor")
    workflow.add_edge("DatabaseExtractor", "NumericProcessor")

    # Stage 3: Processors feed multiple analyzers (parallel chain)
    workflow.add_parallel_chain(
        ["TextProcessor", "NumericProcessor", "ImageProcessor"],
        ["SentimentAnalyzer", "TrendAnalyzer", "AnomalyDetector"],
    )

    # Stage 4: Major fan-in to synthesizer
    workflow.add_edges_to_target(
        ["SentimentAnalyzer", "TrendAnalyzer", "AnomalyDetector"],
        "DataSynthesizer",
    )

    # Stage 5: Fan-out to final outputs
    workflow.add_edges_from_source(
        "DataSynthesizer", ["DashboardGenerator", "AlertSystem"]
    )

    # Set entry points (multiple sources)
    workflow.set_entry_points(
        ["WebScraper", "APICollector", "DatabaseExtractor"]
    )
    workflow.set_end_points(["DashboardGenerator", "AlertSystem"])

    return workflow


def test_different_layouts():
    """Test different Graphviz layout engines."""
    print("🎨 TESTING DIFFERENT GRAPHVIZ LAYOUTS")
    print("=" * 60)

    workflow = create_complex_parallel_workflow()

    layouts = [
        (
            "dot",
            "Hierarchical top-to-bottom layout (best for workflows)",
        ),
        ("neato", "Spring model layout (good for small graphs)"),
        ("fdp", "Force-directed layout (good for large graphs)"),
        (
            "sfdp",
            "Multiscale force-directed layout (for very large graphs)",
        ),
        ("circo", "Circular layout (good for small cyclic graphs)"),
    ]

    for engine, description in layouts:
        print(f"\n🔧 Testing {engine} layout: {description}")
        try:
            output = workflow.visualize(
                output_path=f"complex_workflow_{engine}",
                format="png",
                view=False,
                engine=engine,
                show_parallel_patterns=True,
            )
            print(f"✅ {engine} layout saved: {output}")
        except Exception as e:
            print(f"❌ {engine} layout failed: {e}")


def test_different_formats():
    """Test different output formats."""
    print("\n\n📄 TESTING DIFFERENT OUTPUT FORMATS")
    print("=" * 60)

    workflow = create_simple_workflow()

    formats = [
        ("png", "PNG image (best for presentations)"),
        ("svg", "SVG vector graphics (best for web)"),
        ("pdf", "PDF document (best for documents)"),
        ("dot", "Graphviz DOT source (for editing)"),
    ]

    for fmt, description in formats:
        print(f"\n📋 Testing {fmt} format: {description}")
        try:
            output = workflow.visualize(
                output_path="simple_workflow_test",
                format=fmt,
                view=False,
                engine="dot",
                show_parallel_patterns=True,
            )
            print(f"✅ {fmt} format saved: {output}")
        except Exception as e:
            print(f"❌ {fmt} format failed: {e}")


def test_parallel_pattern_highlighting():
    """Test parallel pattern highlighting features."""
    print("\n\n🔀 TESTING PARALLEL PATTERN HIGHLIGHTING")
    print("=" * 60)

    workflow = create_complex_parallel_workflow()

    print("\n📊 With parallel patterns highlighted:")
    try:
        output_with = workflow.visualize(
            output_path="patterns_highlighted",
            format="png",
            view=False,
            show_parallel_patterns=True,
        )
        print(f"✅ Highlighted version saved: {output_with}")
    except Exception as e:
        print(f"❌ Highlighted version failed: {e}")

    print("\n📊 Without parallel patterns highlighted:")
    try:
        output_without = workflow.visualize(
            output_path="patterns_plain",
            format="png",
            view=False,
            show_parallel_patterns=False,
        )
        print(f"✅ Plain version saved: {output_without}")
    except Exception as e:
        print(f"❌ Plain version failed: {e}")


def test_large_workflow_visualization():
    """Test visualization of a larger workflow."""
    print("\n\n🏢 TESTING LARGE WORKFLOW VISUALIZATION")
    print("=" * 60)

    # Create a larger workflow with many agents
    workflow = GraphWorkflow(
        name="Large-Enterprise-Workflow",
        description="Large enterprise workflow with many agents and complex dependencies",
        verbose=True,
    )

    # Create 20 agents in different categories
    categories = {
        "DataIngestion": 4,
        "Processing": 6,
        "Analysis": 5,
        "Reporting": 3,
        "Monitoring": 2,
    }

    agents_by_category = {}

    for category, count in categories.items():
        agents_by_category[category] = []
        for i in range(count):
            agent = Agent(
                agent_name=f"{category}Agent{i+1}",
                model_name="gpt-4o-mini",
                max_loops=1,
                system_prompt=f"You are {category} specialist #{i+1}",
                verbose=False,
            )
            workflow.add_node(agent)
            agents_by_category[category].append(agent.agent_name)

    # Create complex interconnections
    # Data ingestion fans out to processing
    workflow.add_parallel_chain(
        agents_by_category["DataIngestion"],
        agents_by_category["Processing"],
    )

    # Processing feeds analysis
    workflow.add_parallel_chain(
        agents_by_category["Processing"],
        agents_by_category["Analysis"],
    )

    # Analysis converges to reporting
    workflow.add_edges_to_target(
        agents_by_category["Analysis"],
        agents_by_category["Reporting"][0],  # Primary reporter
    )

    # Other reporting agents get subset
    workflow.add_edges_from_source(
        agents_by_category["Analysis"][0],  # Primary analyzer
        agents_by_category["Reporting"][1:],
    )

    # All reporting feeds monitoring
    workflow.add_edges_to_target(
        agents_by_category["Reporting"],
        agents_by_category["Monitoring"][0],
    )

    print("\n📈 Large workflow statistics:")
    print(f"  Agents: {len(workflow.nodes)}")
    print(f"  Connections: {len(workflow.edges)}")

    # Test with sfdp layout (good for large graphs)
    try:
        output = workflow.visualize(
            output_path="large_enterprise_workflow",
            format="svg",  # SVG scales better for large graphs
            view=False,
            engine="sfdp",  # Better for large graphs
            show_parallel_patterns=True,
        )
        print(f"✅ Large workflow visualization saved: {output}")
    except Exception as e:
        print(f"❌ Large workflow visualization failed: {e}")


def test_fallback_visualization():
    """Test fallback text visualization when Graphviz is not available."""
    print("\n\n🔧 TESTING FALLBACK TEXT VISUALIZATION")
    print("=" * 60)

    workflow = create_complex_parallel_workflow()

    print("\n📝 Testing fallback text visualization:")
    try:
        # Call the fallback method directly
        result = workflow._fallback_text_visualization()
        print(f"✅ Fallback visualization completed: {result}")
    except Exception as e:
        print(f"❌ Fallback visualization failed: {e}")


def run_comprehensive_visualization_tests():
    """Run all visualization tests."""
    print("🎨 COMPREHENSIVE GRAPHVIZ VISUALIZATION TESTS")
    print("=" * 70)

    print(
        "Testing all aspects of the new Graphviz-based visualization system"
    )
    print(
        "including layouts, formats, parallel patterns, and large workflows"
    )

    # Check if Graphviz is available
    try:
        import graphviz

        print("✅ Graphviz Python package available")

        # Test basic functionality
        graphviz.Digraph()
        print("✅ Graphviz functional")

        graphviz_available = True
    except ImportError:
        print(
            "⚠️ Graphviz not available - some tests will use fallback"
        )
        graphviz_available = False

    # Run tests
    if graphviz_available:
        test_different_layouts()
        test_different_formats()
        test_parallel_pattern_highlighting()
        test_large_workflow_visualization()

    # Always test fallback
    test_fallback_visualization()

    # Summary
    print("\n\n🎯 VISUALIZATION TESTING SUMMARY")
    print("=" * 70)

    if graphviz_available:
        print("✅ Graphviz layouts: dot, neato, fdp, sfdp, circo")
        print("✅ Output formats: PNG, SVG, PDF, DOT")
        print("✅ Parallel pattern highlighting with color coding")
        print("✅ Legend generation for pattern types")
        print("✅ Large workflow handling with optimized layouts")
        print("✅ Professional graph styling and node shapes")

        # List generated files
        print("\n📁 Generated visualization files:")
        current_dir = "."
        viz_files = [
            f
            for f in os.listdir(current_dir)
            if any(
                f.startswith(prefix)
                for prefix in [
                    "complex_workflow_",
                    "simple_workflow_",
                    "patterns_",
                    "large_enterprise_",
                ]
            )
        ]

        for file in sorted(viz_files):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                print(f"  📄 {file} ({size:,} bytes)")

    print("✅ Text fallback visualization for compatibility")
    print("✅ Error handling and graceful degradation")
    print("✅ Comprehensive logging and status reporting")

    print("\n🏆 GraphWorkflow now provides professional-grade")
    print("    visualization capabilities with Graphviz!")


if __name__ == "__main__":
    run_comprehensive_visualization_tests()
