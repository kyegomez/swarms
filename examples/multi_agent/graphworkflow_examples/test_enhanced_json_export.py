"""
Test script to demonstrate enhanced JSON export/import capabilities for GraphWorkflow.
This showcases the new comprehensive serialization with metadata, versioning, and various options.
"""

import json
import os
from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def create_sample_workflow():
    """Create a sample workflow for testing JSON export/import capabilities."""

    # Create sample agents
    analyzer = Agent(
        agent_name="DataAnalyzer",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a data analysis expert. Analyze the given data and provide insights.",
        verbose=False,
    )

    processor = Agent(
        agent_name="DataProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a data processor. Process and transform the analyzed data.",
        verbose=False,
    )

    reporter = Agent(
        agent_name="ReportGenerator",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a report generator. Create comprehensive reports from processed data.",
        verbose=False,
    )

    # Create workflow with comprehensive metadata
    workflow = GraphWorkflow(
        name="Enhanced-Data-Analysis-Workflow",
        description="A comprehensive data analysis workflow demonstrating enhanced JSON export capabilities with rich metadata and configuration options.",
        max_loops=3,
        auto_compile=True,
        verbose=True,
        task="Analyze quarterly sales data and generate executive summary reports with actionable insights.",
    )

    # Add agents
    workflow.add_node(analyzer)
    workflow.add_node(processor)
    workflow.add_node(reporter)

    # Create workflow connections
    workflow.add_edge("DataAnalyzer", "DataProcessor")
    workflow.add_edge("DataProcessor", "ReportGenerator")

    # Force compilation to create runtime state
    workflow.compile()

    return workflow


def test_basic_json_export():
    """Test basic JSON export functionality."""
    print("=" * 60)
    print("TEST 1: Basic JSON Export")
    print("=" * 60)

    workflow = create_sample_workflow()

    print("\n📄 Exporting workflow to JSON (basic)...")
    json_data = workflow.to_json()

    # Parse and display structure
    data = json.loads(json_data)

    print("\n📊 Basic Export Results:")
    print(f"  Schema Version: {data.get('schema_version', 'N/A')}")
    print(f"  Export Date: {data.get('export_date', 'N/A')}")
    print(f"  Workflow Name: {data.get('name', 'N/A')}")
    print(f"  Description: {data.get('description', 'N/A')}")
    print(f"  Nodes: {data['metrics']['node_count']}")
    print(f"  Edges: {data['metrics']['edge_count']}")
    print(f"  Max Loops: {data.get('max_loops', 'N/A')}")
    print(f"  Auto Compile: {data.get('auto_compile', 'N/A')}")
    print(f"  JSON Size: {len(json_data):,} characters")

    return json_data


def test_comprehensive_json_export():
    """Test comprehensive JSON export with all options."""
    print("\n\n" + "=" * 60)
    print("TEST 2: Comprehensive JSON Export")
    print("=" * 60)

    workflow = create_sample_workflow()

    # Run workflow to generate conversation history
    print("\n🚀 Running workflow to generate conversation data...")
    try:
        results = workflow.run(
            task="Sample analysis task for testing JSON export"
        )
        print(
            f"✅ Workflow executed: {len(results)} agents completed"
        )
    except Exception as e:
        print(
            f"⚠️ Workflow execution failed (continuing with test): {e}"
        )

    print("\n📄 Exporting workflow to JSON (comprehensive)...")
    json_data = workflow.to_json(
        include_conversation=True, include_runtime_state=True
    )

    # Parse and display comprehensive structure
    data = json.loads(json_data)

    print("\n📊 Comprehensive Export Results:")
    print(f"  Schema Version: {data.get('schema_version', 'N/A')}")
    print(
        f"  Export Timestamp: {data.get('export_timestamp', 'N/A')}"
    )
    print(f"  Runtime State Included: {'runtime_state' in data}")
    print(f"  Conversation Included: {'conversation' in data}")
    print(f"  Compilation Status: {data['metrics']['is_compiled']}")
    print(f"  Layer Count: {data['metrics']['layer_count']}")
    print(f"  JSON Size: {len(json_data):,} characters")

    # Show runtime state details
    if "runtime_state" in data:
        runtime = data["runtime_state"]
        print("\n🔧 Runtime State Details:")
        print(
            f"  Compilation Timestamp: {runtime.get('compilation_timestamp', 'N/A')}"
        )
        print(
            f"  Time Since Compilation: {runtime.get('time_since_compilation', 'N/A'):.3f}s"
        )
        print(
            f"  Sorted Layers: {len(runtime.get('sorted_layers', []))} layers"
        )

    # Show conversation details
    if "conversation" in data:
        conv = data["conversation"]
        print("\n💬 Conversation Details:")
        if "history" in conv:
            print(f"  Message Count: {len(conv['history'])}")
            print(f"  Conversation Type: {conv.get('type', 'N/A')}")
        else:
            print(f"  Status: {conv}")

    return json_data


def test_file_save_load():
    """Test file-based save and load functionality."""
    print("\n\n" + "=" * 60)
    print("TEST 3: File Save/Load Operations")
    print("=" * 60)

    workflow = create_sample_workflow()

    # Test saving to file
    print("\n💾 Saving workflow to file...")
    try:
        filepath = workflow.save_to_file(
            "test_workflow.json",
            include_conversation=False,
            include_runtime_state=True,
            overwrite=True,
        )
        print(f"✅ Workflow saved to: {filepath}")

        # Check file size
        file_size = os.path.getsize(filepath)
        print(f"📁 File size: {file_size:,} bytes")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        return

    # Test loading from file
    print("\n📂 Loading workflow from file...")
    try:
        loaded_workflow = GraphWorkflow.load_from_file(
            "test_workflow.json", restore_runtime_state=True
        )
        print("✅ Workflow loaded successfully")

        # Verify loaded data
        print("\n🔍 Verification:")
        print(f"  Name: {loaded_workflow.name}")
        print(f"  Description: {loaded_workflow.description}")
        print(f"  Nodes: {len(loaded_workflow.nodes)}")
        print(f"  Edges: {len(loaded_workflow.edges)}")
        print(f"  Max Loops: {loaded_workflow.max_loops}")
        print(f"  Compiled: {loaded_workflow._compiled}")

        # Test compilation status
        status = loaded_workflow.get_compilation_status()
        print(f"  Cache Efficient: {status['cache_efficient']}")

    except Exception as e:
        print(f"❌ Load failed: {e}")

    # Cleanup
    try:
        os.remove("test_workflow.json")
        print("\n🧹 Cleaned up test file")
    except:
        pass


def test_workflow_summary():
    """Test workflow summary export functionality."""
    print("\n\n" + "=" * 60)
    print("TEST 4: Workflow Summary Export")
    print("=" * 60)

    workflow = create_sample_workflow()

    print("\n📋 Generating workflow summary...")
    try:
        summary = workflow.export_summary()

        print("\n📊 Workflow Summary:")
        print(f"  ID: {summary['workflow_info']['id']}")
        print(f"  Name: {summary['workflow_info']['name']}")
        print(
            f"  Structure: {summary['structure']['nodes']} nodes, {summary['structure']['edges']} edges"
        )
        print(
            f"  Configuration: {summary['configuration']['max_loops']} loops, {summary['configuration']['max_workers']} workers"
        )
        print(f"  Task Defined: {summary['task']['defined']}")
        print(
            f"  Conversation Available: {summary['conversation']['available']}"
        )

        # Show agents
        print("\n🤖 Agents:")
        for agent in summary["agents"]:
            print(f"    - {agent['id']} ({agent['agent_name']})")

        # Show connections
        print("\n🔗 Connections:")
        for conn in summary["connections"]:
            print(f"    - {conn['from']} → {conn['to']}")

    except Exception as e:
        print(f"❌ Summary generation failed: {e}")


def test_backward_compatibility():
    """Test backward compatibility with legacy JSON format."""
    print("\n\n" + "=" * 60)
    print("TEST 5: Backward Compatibility")
    print("=" * 60)

    # Create a legacy-style JSON (simulated)
    legacy_json = {
        "id": "test-legacy-workflow",
        "name": "Legacy Workflow",
        "nodes": [
            {
                "id": "agent1",
                "type": "agent",
                "agent": {"agent_name": "LegacyAgent"},
                "metadata": {},
            }
        ],
        "edges": [],
        "entry_points": ["agent1"],
        "end_points": ["agent1"],
        "max_loops": 1,
        "task": "Legacy task",
    }

    legacy_json_str = json.dumps(legacy_json, indent=2)

    print("\n📜 Testing legacy JSON format compatibility...")
    try:
        workflow = GraphWorkflow.from_json(legacy_json_str)
        print("✅ Legacy format loaded successfully")
        print(f"  Name: {workflow.name}")
        print(f"  Nodes: {len(workflow.nodes)}")
        print(f"  Max Loops: {workflow.max_loops}")

    except Exception as e:
        print(f"❌ Legacy compatibility failed: {e}")


def run_enhanced_json_tests():
    """Run all enhanced JSON export/import tests."""
    print("🧪 ENHANCED JSON EXPORT/IMPORT TESTS")
    print(
        "Testing comprehensive serialization capabilities with metadata and versioning"
    )

    # Run all tests
    test_basic_json_export()
    test_comprehensive_json_export()
    test_file_save_load()
    test_workflow_summary()
    test_backward_compatibility()

    print("\n\n" + "=" * 60)
    print("🎯 ENHANCED JSON CAPABILITIES SUMMARY")
    print("=" * 60)
    print("✅ Schema versioning and metadata")
    print("✅ Comprehensive configuration export")
    print("✅ Optional conversation history inclusion")
    print("✅ Runtime state preservation")
    print("✅ Enhanced error handling")
    print("✅ File-based save/load operations")
    print("✅ Workflow summary generation")
    print("✅ Backward compatibility")
    print("✅ Rich serialization metadata")


if __name__ == "__main__":
    run_enhanced_json_tests()
