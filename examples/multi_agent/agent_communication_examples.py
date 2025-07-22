"""
Agent Multi-Agent Communication Examples

This file demonstrates the multi-agent communication methods available in the Agent class:
- talk_to: Direct communication between two agents
- talk_to_multiple_agents: Concurrent communication with multiple agents
- receive_message: Process incoming messages from other agents
- send_agent_message: Send formatted messages to other agents

Run: python agent_communication_examples.py
"""

import os
from swarms import Agent

# Set up your API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


def example_1_direct_agent_communication():
    """Example 1: Direct communication between two agents using talk_to method"""
    print("=" * 60)
    print("Example 1: Direct Agent Communication")
    print("=" * 60)

    # Create two specialized agents
    researcher = Agent(
        agent_name="Research-Agent",
        system_prompt="You are a research specialist focused on gathering and analyzing information. Provide detailed, fact-based responses.",
        max_loops=1,
        verbose=False,
    )

    analyst = Agent(
        agent_name="Analysis-Agent",
        system_prompt="You are an analytical specialist focused on interpreting research data and providing strategic insights.",
        max_loops=1,
        verbose=False,
    )

    # Agent communication
    print("Researcher talking to Analyst...")
    research_result = researcher.talk_to(
        agent=analyst,
        task="Analyze the market trends for renewable energy stocks and provide investment recommendations",
    )

    print(f"\nFinal Analysis Result:\n{research_result}")
    return research_result


def example_2_multiple_agent_communication():
    """Example 2: Broadcasting to multiple agents using talk_to_multiple_agents"""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Agent Communication")
    print("=" * 60)

    # Create multiple specialized agents
    agents = [
        Agent(
            agent_name="Financial-Analyst",
            system_prompt="You are a financial analysis expert specializing in stock valuation and market trends.",
            max_loops=1,
            verbose=False,
        ),
        Agent(
            agent_name="Risk-Assessor",
            system_prompt="You are a risk assessment specialist focused on identifying potential investment risks.",
            max_loops=1,
            verbose=False,
        ),
        Agent(
            agent_name="Market-Researcher",
            system_prompt="You are a market research expert specializing in industry analysis and competitive intelligence.",
            max_loops=1,
            verbose=False,
        ),
    ]

    coordinator = Agent(
        agent_name="Coordinator-Agent",
        system_prompt="You coordinate multi-agent analysis and synthesize diverse perspectives into actionable insights.",
        max_loops=1,
        verbose=False,
    )

    # Broadcast to multiple agents
    print("Coordinator broadcasting to multiple agents...")
    responses = coordinator.talk_to_multiple_agents(
        agents=agents,
        task="Evaluate the investment potential of Tesla stock for the next quarter",
    )

    # Process responses
    print("\nResponses from all agents:")
    for i, response in enumerate(responses):
        if response:
            print(f"\n{agents[i].agent_name} Response:")
            print("-" * 40)
            print(
                response[:200] + "..."
                if len(response) > 200
                else response
            )
        else:
            print(f"\n{agents[i].agent_name}: Failed to respond")

    return responses


def example_3_message_handling():
    """Example 3: Message handling using receive_message and send_agent_message"""
    print("\n" + "=" * 60)
    print("Example 3: Message Handling")
    print("=" * 60)

    # Create an agent that can receive messages
    support_agent = Agent(
        agent_name="Support-Agent",
        system_prompt="You provide helpful support and assistance. Always be professional and solution-oriented.",
        max_loops=1,
        verbose=False,
    )

    notification_agent = Agent(
        agent_name="Notification-Agent",
        system_prompt="You send notifications and updates to other systems and agents.",
        max_loops=1,
        verbose=False,
    )

    # Example of receiving a message
    print("Support agent receiving message...")
    received_response = support_agent.receive_message(
        agent_name="Customer-Service-Agent",
        task="A customer is asking about our refund policies for software purchases. Can you provide guidance?",
    )
    print(f"\nSupport Agent Response:\n{received_response}")

    # Example of sending a message
    print("\nNotification agent sending message...")
    sent_result = notification_agent.send_agent_message(
        agent_name="Task-Manager-Agent",
        message="Customer support ticket #12345 has been resolved successfully",
    )
    print(f"\nNotification Result:\n{sent_result}")

    return received_response, sent_result


def example_4_sequential_workflow():
    """Example 4: Sequential agent workflow using communication methods"""
    print("\n" + "=" * 60)
    print("Example 4: Sequential Agent Workflow")
    print("=" * 60)

    # Create specialized agents for a document processing workflow
    extractor = Agent(
        agent_name="Data-Extractor",
        system_prompt="You extract key information and data points from documents. Focus on accuracy and completeness.",
        max_loops=1,
        verbose=False,
    )

    validator = Agent(
        agent_name="Data-Validator",
        system_prompt="You validate and verify extracted data for accuracy, completeness, and consistency.",
        max_loops=1,
        verbose=False,
    )

    formatter = Agent(
        agent_name="Data-Formatter",
        system_prompt="You format validated data into structured, professional reports and summaries.",
        max_loops=1,
        verbose=False,
    )

    # Sequential processing workflow
    document_content = """
    Q3 Financial Report Summary:
    - Revenue: $2.5M (up 15% from Q2)
    - Expenses: $1.8M (operational costs increased by 8%)
    - Net Profit: $700K (improved profit margin of 28%)
    - New Customers: 1,200 (25% growth rate)
    - Customer Retention: 92%
    - Market Share: Increased to 12% in our sector
    """

    print("Starting sequential workflow...")

    # Step 1: Extract data
    print("\nStep 1: Data Extraction")
    extracted_data = extractor.run(
        f"Extract key financial metrics from this report: {document_content}"
    )
    print(f"Extracted: {extracted_data[:150]}...")

    # Step 2: Validate data
    print("\nStep 2: Data Validation")
    validated_data = extractor.talk_to(
        agent=validator,
        task=f"Please validate this extracted data for accuracy and completeness: {extracted_data}",
    )
    print(f"Validated: {validated_data[:150]}...")

    # Step 3: Format data
    print("\nStep 3: Data Formatting")
    final_output = validator.talk_to(
        agent=formatter,
        task=f"Format this validated data into a structured executive summary: {validated_data}",
    )

    print(f"\nFinal Report:\n{final_output}")
    return final_output


def example_5_error_handling():
    """Example 5: Robust communication with error handling"""
    print("\n" + "=" * 60)
    print("Example 5: Communication with Error Handling")
    print("=" * 60)

    def safe_agent_communication(sender, receiver, message):
        """Safely handle agent communication with comprehensive error handling"""
        try:
            print(
                f"Attempting communication: {sender.agent_name} -> {receiver.agent_name}"
            )
            response = sender.talk_to(agent=receiver, task=message)
            return {
                "success": True,
                "response": response,
                "error": None,
            }
        except Exception as e:
            print(f"Communication failed: {e}")
            return {
                "success": False,
                "response": None,
                "error": str(e),
            }

    # Create agents
    agent_a = Agent(
        agent_name="Agent-A",
        system_prompt="You are a helpful assistant focused on providing accurate information.",
        max_loops=1,
        verbose=False,
    )

    agent_b = Agent(
        agent_name="Agent-B",
        system_prompt="You are a knowledgeable expert in technology and business trends.",
        max_loops=1,
        verbose=False,
    )

    # Safe communication
    result = safe_agent_communication(
        sender=agent_a,
        receiver=agent_b,
        message="What are the latest trends in artificial intelligence and how might they impact business operations?",
    )

    if result["success"]:
        print("\nCommunication successful!")
        print(f"Response: {result['response'][:200]}...")
    else:
        print(f"\nCommunication failed: {result['error']}")

    return result


def main():
    """Run all multi-agent communication examples"""
    print("🤖 Agent Multi-Agent Communication Examples")
    print(
        "This demonstrates the communication methods available in the Agent class"
    )

    try:
        # Run all examples
        example_1_direct_agent_communication()
        example_2_multiple_agent_communication()
        example_3_message_handling()
        example_4_sequential_workflow()
        example_5_error_handling()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print(
            "Make sure to set your OPENAI_API_KEY environment variable"
        )


if __name__ == "__main__":
    main()
