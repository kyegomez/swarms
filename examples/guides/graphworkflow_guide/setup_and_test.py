#!/usr/bin/env python3
"""
GraphWorkflow Setup and Test Script
==================================

This script helps you set up and test your GraphWorkflow environment.
It checks dependencies, validates the installation, and runs basic tests.

Usage:
    python setup_and_test.py [--install-deps] [--run-tests] [--check-only]
"""

import sys
import subprocess
import importlib
import argparse
from typing import Dict, List, Tuple


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")

    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(
            f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible"
        )
        return True
    else:
        print(
            f"❌ Python {version.major}.{version.minor}.{version.micro} is too old"
        )
        print("   GraphWorkflow requires Python 3.8 or newer")
        return False


def check_package_installation(
    package: str, import_name: str = None
) -> bool:
    """Check if a package is installed and importable."""
    import_name = import_name or package

    try:
        importlib.import_module(import_name)
        print(f"✅ {package} is installed and importable")
        return True
    except ImportError:
        print(f"❌ {package} is not installed or not importable")
        return False


def install_package(package: str) -> bool:
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}")
        print(f"   Error: {e.stderr}")
        return False


def check_core_dependencies() -> Dict[str, bool]:
    """Check core dependencies required for GraphWorkflow."""
    print("\n🔍 Checking core dependencies...")

    dependencies = {
        "swarms": "swarms",
        "networkx": "networkx",
    }

    results = {}
    for package, import_name in dependencies.items():
        results[package] = check_package_installation(
            package, import_name
        )

    return results


def check_optional_dependencies() -> Dict[str, bool]:
    """Check optional dependencies for enhanced features."""
    print("\n🔍 Checking optional dependencies...")

    optional_deps = {
        "graphviz": "graphviz",
        "psutil": "psutil",
    }

    results = {}
    for package, import_name in optional_deps.items():
        results[package] = check_package_installation(
            package, import_name
        )

    return results


def test_basic_import() -> bool:
    """Test basic GraphWorkflow import."""
    print("\n🧪 Testing basic GraphWorkflow import...")

    try:
        from swarms.structs.graph_workflow import GraphWorkflow

        print("✅ GraphWorkflow imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import GraphWorkflow: {e}")
        return False


def test_agent_import() -> bool:
    """Test Agent import."""
    print("\n🧪 Testing Agent import...")

    try:
        from swarms import Agent

        print("✅ Agent imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Agent: {e}")
        return False


def test_basic_workflow_creation() -> bool:
    """Test basic workflow creation."""
    print("\n🧪 Testing basic workflow creation...")

    try:
        from swarms import Agent
        from swarms.structs.graph_workflow import GraphWorkflow

        # Create a simple agent
        agent = Agent(
            agent_name="TestAgent",
            model_name="gpt-4o-mini",
            max_loops=1,
            system_prompt="You are a test agent.",
            verbose=False,
        )

        # Create workflow
        workflow = GraphWorkflow(
            name="TestWorkflow",
            description="A test workflow",
            verbose=False,
            auto_compile=True,
        )

        # Add agent
        workflow.add_node(agent)

        print("✅ Basic workflow creation successful")
        print(f"   Created workflow with {len(workflow.nodes)} nodes")
        return True

    except Exception as e:
        print(f"❌ Basic workflow creation failed: {e}")
        return False


def test_workflow_compilation() -> bool:
    """Test workflow compilation."""
    print("\n🧪 Testing workflow compilation...")

    try:
        from swarms import Agent
        from swarms.structs.graph_workflow import GraphWorkflow

        # Create agents
        agent1 = Agent(
            agent_name="Agent1",
            model_name="gpt-4o-mini",
            max_loops=1,
            system_prompt="You are agent 1.",
            verbose=False,
        )

        agent2 = Agent(
            agent_name="Agent2",
            model_name="gpt-4o-mini",
            max_loops=1,
            system_prompt="You are agent 2.",
            verbose=False,
        )

        # Create workflow
        workflow = GraphWorkflow(
            name="CompilationTestWorkflow",
            description="A workflow for testing compilation",
            verbose=False,
            auto_compile=False,  # Manual compilation
        )

        # Add agents and edges
        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_edge("Agent1", "Agent2")

        # Test compilation
        workflow.compile()

        # Check compilation status
        status = workflow.get_compilation_status()

        if status["is_compiled"]:
            print("✅ Workflow compilation successful")
            print(
                f"   Layers: {status.get('cached_layers_count', 'N/A')}"
            )
            print(f"   Workers: {status.get('max_workers', 'N/A')}")
            return True
        else:
            print("❌ Workflow compilation failed - not compiled")
            return False

    except Exception as e:
        print(f"❌ Workflow compilation failed: {e}")
        return False


def test_workflow_validation() -> bool:
    """Test workflow validation."""
    print("\n🧪 Testing workflow validation...")

    try:
        from swarms import Agent
        from swarms.structs.graph_workflow import GraphWorkflow

        # Create a simple workflow
        agent = Agent(
            agent_name="ValidationTestAgent",
            model_name="gpt-4o-mini",
            max_loops=1,
            system_prompt="You are a validation test agent.",
            verbose=False,
        )

        workflow = GraphWorkflow(
            name="ValidationTestWorkflow",
            description="A workflow for testing validation",
            verbose=False,
            auto_compile=True,
        )

        workflow.add_node(agent)

        # Test validation
        validation = workflow.validate(auto_fix=True)

        print("✅ Workflow validation successful")
        print(f"   Valid: {validation['is_valid']}")
        print(f"   Warnings: {len(validation['warnings'])}")
        print(f"   Errors: {len(validation['errors'])}")

        return True

    except Exception as e:
        print(f"❌ Workflow validation failed: {e}")
        return False


def test_serialization() -> bool:
    """Test workflow serialization."""
    print("\n🧪 Testing workflow serialization...")

    try:
        from swarms import Agent
        from swarms.structs.graph_workflow import GraphWorkflow

        # Create a simple workflow
        agent = Agent(
            agent_name="SerializationTestAgent",
            model_name="gpt-4o-mini",
            max_loops=1,
            system_prompt="You are a serialization test agent.",
            verbose=False,
        )

        workflow = GraphWorkflow(
            name="SerializationTestWorkflow",
            description="A workflow for testing serialization",
            verbose=False,
            auto_compile=True,
        )

        workflow.add_node(agent)

        # Test JSON serialization
        json_data = workflow.to_json()

        if len(json_data) > 0:
            print("✅ JSON serialization successful")
            print(f"   JSON size: {len(json_data)} characters")

            # Test deserialization
            restored = GraphWorkflow.from_json(json_data)
            print("✅ JSON deserialization successful")
            print(f"   Restored nodes: {len(restored.nodes)}")

            return True
        else:
            print("❌ JSON serialization failed - empty result")
            return False

    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False


def run_all_tests() -> List[Tuple[str, bool]]:
    """Run all tests and return results."""
    print("\n🚀 Running GraphWorkflow Tests")
    print("=" * 50)

    tests = [
        ("Basic Import", test_basic_import),
        ("Agent Import", test_agent_import),
        ("Basic Workflow Creation", test_basic_workflow_creation),
        ("Workflow Compilation", test_workflow_compilation),
        ("Workflow Validation", test_workflow_validation),
        ("Serialization", test_serialization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    return results


def print_test_summary(results: List[Tuple[str, bool]]):
    """Print test summary."""
    print("\n📊 TEST SUMMARY")
    print("=" * 30)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print("-" * 30)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! GraphWorkflow is ready to use.")
    else:
        print(
            f"\n⚠️ {total-passed} tests failed. Please check the output above."
        )
        print(
            "   Consider running with --install-deps to install missing packages."
        )


def main():
    """Main setup and test function."""
    parser = argparse.ArgumentParser(
        description="GraphWorkflow Setup and Test"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install missing dependencies",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run functionality tests",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies, don't install",
    )

    args = parser.parse_args()

    # If no arguments, run everything
    if not any([args.install_deps, args.run_tests, args.check_only]):
        args.install_deps = True
        args.run_tests = True

    print("🌟 GRAPHWORKFLOW SETUP AND TEST")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print(
            "\n❌ Python version incompatible. Please upgrade Python."
        )
        sys.exit(1)

    # Check dependencies
    core_deps = check_core_dependencies()
    optional_deps = check_optional_dependencies()

    # Install missing dependencies if requested
    if args.install_deps and not args.check_only:
        print("\n📦 Installing missing dependencies...")

        # Install core dependencies
        for package, installed in core_deps.items():
            if not installed:
                if not install_package(package):
                    print(
                        f"\n❌ Failed to install core dependency: {package}"
                    )
                    sys.exit(1)

        # Install optional dependencies
        for package, installed in optional_deps.items():
            if not installed:
                print(
                    f"\n📦 Installing optional dependency: {package}"
                )
                install_package(
                    package
                )  # Don't fail on optional deps

    # Run tests if requested
    if args.run_tests:
        results = run_all_tests()
        print_test_summary(results)

        # Exit with error code if tests failed
        failed_tests = sum(1 for _, result in results if not result)
        if failed_tests > 0:
            sys.exit(1)

    elif args.check_only:
        # Summary for check-only mode
        core_missing = sum(
            1 for installed in core_deps.values() if not installed
        )
        optional_missing = sum(
            1 for installed in optional_deps.values() if not installed
        )

        print("\n📊 DEPENDENCY CHECK SUMMARY")
        print("=" * 40)
        print(f"Core dependencies missing: {core_missing}")
        print(f"Optional dependencies missing: {optional_missing}")

        if core_missing > 0:
            print(
                "\n⚠️ Missing core dependencies. Run with --install-deps to install."
            )
            sys.exit(1)
        else:
            print("\n✅ All core dependencies satisfied!")

    print("\n🎯 Next Steps:")
    print("1. Run the quick start guide: python quick_start_guide.py")
    print(
        "2. Try the comprehensive demo: python comprehensive_demo.py"
    )
    print("3. Explore healthcare and finance examples")
    print("4. Read the technical documentation")


if __name__ == "__main__":
    main()
