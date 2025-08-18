"""
Single-File Hierarchical Structured Communication Framework Example

This example demonstrates how to use the consolidated single-file implementation
of the Talk Structurally, Act Hierarchically framework.

All components are now in one file: hierarchical_structured_communication_framework.py
"""

import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

# Import everything from the single file
from swarms.structs.hierarchical_structured_communication_framework import (
    HierarchicalStructuredCommunicationFramework,
    HierarchicalStructuredCommunicationGenerator,
    HierarchicalStructuredCommunicationEvaluator,
    HierarchicalStructuredCommunicationRefiner,
    HierarchicalStructuredCommunicationSupervisor,
    # Convenience aliases
    TalkHierarchicalGenerator,
    TalkHierarchicalEvaluator,
    TalkHierarchicalRefiner,
    TalkHierarchicalSupervisor,
)

# Load environment variables
load_dotenv()


def example_basic_usage():
    """
    Basic usage example with default agents
    """
    print("=" * 80)
    print("BASIC USAGE EXAMPLE")
    print("=" * 80)
    
    # Create framework with default configuration
    framework = HierarchicalStructuredCommunicationFramework(
        name="BasicFramework",
        max_loops=2,
        verbose=True
    )
    
    # Run a simple task
    task = "Explain the benefits of structured communication in multi-agent systems"
    
    print(f"Task: {task}")
    print("Running framework...")
    
    result = framework.run(task)
    
    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(result["final_result"])
    
    print(f"\nTotal loops: {result['total_loops']}")
    print(f"Conversation history entries: {len(result['conversation_history'])}")
    print(f"Evaluation results: {len(result['evaluation_results'])}")


def example_custom_agents():
    """
    Example using custom specialized agents
    """
    print("\n" + "=" * 80)
    print("CUSTOM AGENTS EXAMPLE")
    print("=" * 80)
    
    # Create custom agents using the convenience aliases
    generator = TalkHierarchicalGenerator(
        agent_name="ContentCreator",
        model_name="gpt-4o-mini",
        verbose=True
    )
    
    evaluator1 = TalkHierarchicalEvaluator(
        agent_name="AccuracyChecker",
        evaluation_criteria=["accuracy", "technical_correctness"],
        model_name="gpt-4o-mini",
        verbose=True
    )
    
    evaluator2 = TalkHierarchicalEvaluator(
        agent_name="ClarityChecker",
        evaluation_criteria=["clarity", "readability", "coherence"],
        model_name="gpt-4o-mini",
        verbose=True
    )
    
    refiner = TalkHierarchicalRefiner(
        agent_name="ContentImprover",
        model_name="gpt-4o-mini",
        verbose=True
    )
    
    supervisor = TalkHierarchicalSupervisor(
        agent_name="WorkflowManager",
        model_name="gpt-4o-mini",
        verbose=True
    )
    
    # Create framework with custom agents
    framework = HierarchicalStructuredCommunicationFramework(
        name="CustomFramework",
        supervisor=supervisor,
        generators=[generator],
        evaluators=[evaluator1, evaluator2],
        refiners=[refiner],
        max_loops=3,
        verbose=True
    )
    
    # Run a complex task
    task = "Design a comprehensive machine learning pipeline for sentiment analysis"
    
    print(f"Task: {task}")
    print("Running framework with custom agents...")
    
    result = framework.run(task)
    
    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(result["final_result"])
    
    print(f"\nTotal loops: {result['total_loops']}")
    print(f"Conversation history entries: {len(result['conversation_history'])}")
    print(f"Evaluation results: {len(result['evaluation_results'])}")


def example_ollama_integration():
    """
    Example using Ollama for local inference
    """
    print("\n" + "=" * 80)
    print("OLLAMA INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # Create framework with Ollama configuration
    framework = HierarchicalStructuredCommunicationFramework(
        name="OllamaFramework",
        max_loops=2,
        verbose=True,
        model_name="llama3:latest",
        use_ollama=True,
        ollama_base_url="http://localhost:11434/v1",
        ollama_api_key="ollama"
    )
    
    # Run a task with local model
    task = "Explain the concept of structured communication protocols"
    
    print(f"Task: {task}")
    print("Running framework with Ollama...")
    
    try:
        result = framework.run(task)
        
        print("\n" + "=" * 50)
        print("FINAL RESULT")
        print("=" * 50)
        print(result["final_result"])
        
        print(f"\nTotal loops: {result['total_loops']}")
        print(f"Conversation history entries: {len(result['conversation_history'])}")
        print(f"Evaluation results: {len(result['evaluation_results'])}")
        
    except Exception as e:
        print(f"Error with Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")


def example_structured_communication():
    """
    Example demonstrating structured communication protocol
    """
    print("\n" + "=" * 80)
    print("STRUCTURED COMMUNICATION EXAMPLE")
    print("=" * 80)
    
    # Create framework
    framework = HierarchicalStructuredCommunicationFramework(
        name="CommunicationDemo",
        verbose=True
    )
    
    # Demonstrate structured message sending
    print("Sending structured message...")
    
    structured_msg = framework.send_structured_message(
        sender="Supervisor",
        recipient="Generator",
        message="Create a technical documentation outline",
        background="For a Python library focused on data processing",
        intermediate_output="Previous research on similar libraries"
    )
    
    print(f"Message sent: {structured_msg.message}")
    print(f"Background: {structured_msg.background}")
    print(f"Intermediate output: {structured_msg.intermediate_output}")
    print(f"From: {structured_msg.sender} -> To: {structured_msg.recipient}")


def example_agent_interaction():
    """
    Example showing direct agent interaction
    """
    print("\n" + "=" * 80)
    print("AGENT INTERACTION EXAMPLE")
    print("=" * 80)
    
    # Create agents
    generator = TalkHierarchicalGenerator(
        agent_name="ContentGenerator",
        verbose=True
    )
    
    evaluator = TalkHierarchicalEvaluator(
        agent_name="QualityEvaluator",
        evaluation_criteria=["accuracy", "clarity"],
        verbose=True
    )
    
    refiner = TalkHierarchicalRefiner(
        agent_name="ContentRefiner",
        verbose=True
    )
    
    # Generate content
    print("1. Generating content...")
    gen_result = generator.generate_with_structure(
        message="Create a brief explanation of machine learning",
        background="For beginners with no technical background",
        intermediate_output=""
    )
    
    print(f"Generated content: {gen_result.content[:200]}...")
    
    # Evaluate content
    print("\n2. Evaluating content...")
    eval_result = evaluator.evaluate_with_criterion(
        content=gen_result.content,
        criterion="clarity"
    )
    
    print(f"Evaluation score: {eval_result.score}/10")
    print(f"Feedback: {eval_result.feedback[:200]}...")
    
    # Refine content
    print("\n3. Refining content...")
    refine_result = refiner.refine_with_feedback(
        original_content=gen_result.content,
        evaluation_results=[eval_result]
    )
    
    print(f"Refined content: {refine_result.refined_content[:200]}...")
    print(f"Changes made: {refine_result.changes_made}")


def main():
    """
    Main function to run all examples
    """
    print("SINGLE-FILE HIERARCHICAL STRUCTURED COMMUNICATION FRAMEWORK")
    print("=" * 80)
    print("This demonstrates the consolidated single-file implementation")
    print("based on the research paper: arXiv:2502.11098")
    print("=" * 80)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_agents()
        example_ollama_integration()
        example_structured_communication()
        example_agent_interaction()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Framework Features Demonstrated:")
        print("✓ Single-file implementation")
        print("✓ Structured Communication Protocol (M_ij, B_ij, I_ij)")
        print("✓ Hierarchical Evaluation System")
        print("✓ Iterative Refinement Process")
        print("✓ Flexible Model Configuration (OpenAI/Ollama)")
        print("✓ Custom Agent Specialization")
        print("✓ Direct Agent Interaction")
        print("✓ Convenience Aliases")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
