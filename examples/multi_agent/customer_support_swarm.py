"""
Customer Support Agent Swarm with Vector Database

This example demonstrates a multi-agent customer support system with:
- Console-based interactive customer support
- ChromaDB vector database for company knowledge
- Intelligent routing and agent communication
- Real-time question answering from knowledge base
"""

import chromadb
from chromadb.config import Settings
from swarms import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Initialize ChromaDB client (in-memory, free, no setup required)
chroma_client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    allow_reset=True
))

# Create or get collection for company knowledge
knowledge_collection = chroma_client.get_or_create_collection(
    name="company_knowledge",
    metadata={"description": "TechCorp product information and support knowledge"}
)

# Company knowledge base entries to store in vector DB
KNOWLEDGE_ENTRIES = [
    {
        "id": "product_cloudsync",
        "text": "CloudSync Pro is our entry-level product priced at $29.99/month. It includes unlimited storage, real-time sync across devices, end-to-end encryption for security, and 24/7 support via chat, email, and phone. New users get a 14-day free trial.",
        "metadata": {"category": "product", "product": "CloudSync Pro"}
    },
    {
        "id": "product_datavault",
        "text": "DataVault Enterprise is our premium product priced at $99.99/month. It's designed for teams and includes team collaboration features, admin controls, compliance tools, API access, dedicated account manager, and priority support. New customers get a 30-day free trial.",
        "metadata": {"category": "product", "product": "DataVault Enterprise"}
    },
    {
        "id": "troubleshoot_login",
        "text": "Login problems are common and usually resolved by clearing browser cache and cookies, or using the password reset feature. If issues persist, check if the account is active and verify the email address is correct. Contact support if problems continue.",
        "metadata": {"category": "troubleshooting", "issue": "login"}
    },
    {
        "id": "troubleshoot_sync",
        "text": "Sync delays can occur due to poor internet connection, outdated app versions, or server maintenance. First check internet connectivity, then ensure you're using the latest app version. Sync typically completes within 30 seconds on good connections.",
        "metadata": {"category": "troubleshooting", "issue": "sync"}
    },
    {
        "id": "billing_refund",
        "text": "TechCorp offers a 30-day money-back guarantee on all products. Refund requests can be submitted through the account portal or by emailing billing@techcorp.com. Refunds are processed within 5-7 business days.",
        "metadata": {"category": "billing", "topic": "refund"}
    },
    {
        "id": "billing_upgrade",
        "text": "Upgrading from CloudSync Pro to DataVault Enterprise is easy through the account portal. You'll be charged the prorated difference immediately, and all your data migrates automatically. Downgrades take effect at the next billing cycle.",
        "metadata": {"category": "billing", "topic": "upgrade"}
    },
    {
        "id": "policy_privacy",
        "text": "TechCorp is fully GDPR and SOC 2 compliant. We use end-to-end encryption, never sell customer data, and provide full data export capabilities. Our privacy policy is available at techcorp.com/privacy.",
        "metadata": {"category": "policy", "topic": "privacy"}
    },
    {
        "id": "policy_sla",
        "text": "TechCorp guarantees 99.9% uptime for all services. If we fail to meet this SLA, customers receive service credits. Real-time status is available at status.techcorp.com.",
        "metadata": {"category": "policy", "topic": "sla"}
    },
]

# Populate vector database with company knowledge
print("📚 Loading company knowledge into vector database...")
for entry in KNOWLEDGE_ENTRIES:
    knowledge_collection.add(
        ids=[entry["id"]],
        documents=[entry["text"]],
        metadatas=[entry["metadata"]]
    )
print(f"✅ Loaded {len(KNOWLEDGE_ENTRIES)} knowledge entries into ChromaDB\n")


def query_knowledge_base(query: str, n_results: int = 3) -> str:
    """Query the vector database for relevant knowledge"""
    results = knowledge_collection.query(
        query_texts=[query],
        n_results=n_results
    )

    if not results["documents"][0]:
        return "No relevant information found in knowledge base."

    # Format results
    knowledge = "Relevant Knowledge Base Information:\n\n"
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        knowledge += f"{i}. [{metadata.get('category', 'general')}] {doc}\n\n"

    return knowledge


# Create specialized support agents that use the vector database
def create_support_agents():
    """Create a team of specialized customer support agents with vector DB access"""

    # Agent 1: Triage Agent with Vector DB
    triage_agent = Agent(
        agent_name="Triage-Agent",
        agent_description="First point of contact. Uses vector DB to categorize inquiries.",
        system_prompt="""You are a customer support triage specialist for TechCorp.

Your role is to:
1. Understand the customer's issue or question
2. Categorize it (technical, billing, product info, or general)
3. Extract key information (product, error messages, account details)
4. Provide a brief summary

You have access to a knowledge base that will be provided with relevant information.

Always be empathetic and professional. Keep your response concise.
Format: CATEGORY: [category] | SUMMARY: [brief summary]
""",
        max_loops=1,
        verbose=False,
    )

    # Agent 2: Support Agent with Vector DB
    support_agent = Agent(
        agent_name="Support-Agent",
        agent_description="Handles customer inquiries using knowledge base.",
        system_prompt="""You are a TechCorp support specialist.

You help with:
- Technical issues (login, sync, performance)
- Billing questions (pricing, refunds, upgrades)
- Product information (features, plans, comparisons)

Relevant knowledge base information will be provided to you.

Provide clear, actionable solutions. Be professional and helpful.
Format your response clearly with the solution and any next steps.
""",
        max_loops=1,
        verbose=False,
    )

    return [triage_agent, support_agent]


def handle_support_query(customer_query: str, agents: list) -> str:
    """Handle a customer support query using vector DB + agents"""

    # Step 1: Query vector database for relevant knowledge
    print("\n🔍 Searching knowledge base...")
    relevant_knowledge = query_knowledge_base(customer_query, n_results=3)

    # Step 2: Combine query with knowledge for agent processing
    enriched_query = f"""Customer Query: {customer_query}

{relevant_knowledge}

Based on the customer query and knowledge base information above, provide an appropriate response."""

    # Step 3: Run through agent workflow
    print("🤖 Processing with support agents...\n")

    workflow = SequentialWorkflow(
        name="Support-Workflow",
        description="Customer support with vector DB knowledge",
        agents=agents,
        max_loops=1,
    )

    result = workflow.run(enriched_query)
    return result


def interactive_console():
    """Run interactive console-based customer support"""
    print("\n" + "=" * 80)
    print("🎯 TECHCORP CUSTOMER SUPPORT - INTERACTIVE CONSOLE")
    print("=" * 80)
    print("\nPowered by:")
    print("  • Multi-Agent System (Swarms)")
    print("  • ChromaDB Vector Database (free, local)")
    print("  • Real-time Knowledge Retrieval")
    print("\nType 'quit' or 'exit' to end the session")
    print("Type 'help' to see example questions")
    print("=" * 80 + "\n")

    # Create agents once
    agents = create_support_agents()

    while True:
        # Get user input
        print("\n" + "-" * 80)
        user_input = input("\n👤 You: ").strip()

        if not user_input:
            continue

        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            print("\n👋 Thank you for contacting TechCorp Support! Have a great day!")
            break

        if user_input.lower() == 'help':
            print("\n📋 Example questions you can ask:")
            print("  • What's the difference between CloudSync Pro and DataVault Enterprise?")
            print("  • I can't log into my account, what should I do?")
            print("  • How do I upgrade my subscription?")
            print("  • What's your refund policy?")
            print("  • My files aren't syncing, can you help?")
            continue

        # Process support query
        try:
            handle_support_query(user_input, agents)

        except Exception as e:
            print(f"\n❌ Error processing query: {str(e)}")
            print("Please try rephrasing your question or contact support@techcorp.com")


def demo_mode():
    """Run demo mode with pre-defined queries"""
    print("\n" + "=" * 80)
    print("🎯 TECHCORP CUSTOMER SUPPORT - DEMO MODE")
    print("=" * 80 + "\n")

    agents = create_support_agents()

    demo_queries = [
        "I can't log into CloudSync Pro. I tried resetting my password but still getting errors.",
        "What's the difference between CloudSync Pro and DataVault Enterprise? Which is better for a team of 10?",
        "I was charged twice this month. Can I get a refund?",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"DEMO QUERY {i}/{len(demo_queries)}")
        print(f"{'='*80}")
        print(f"\n👤 Customer: {query}\n")

        handle_support_query(query, agents)

        if i < len(demo_queries):
            input("\nPress Enter to continue to next query...")


if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_mode()
    else:
        interactive_console()
