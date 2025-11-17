#!/usr/bin/env python3
"""
ForestSwarm Example Script

This script demonstrates the ForestSwarm functionality with realistic examples
of financial services and investment management agents.
"""

from swarms.structs.tree_swarm import TreeAgent, Tree, ForestSwarm


def create_financial_services_forest():
    """Create a comprehensive financial services forest with multiple specialized agents."""

    print("🌳 Creating Financial Services Forest...")

    # Financial Services Tree - Personal Finance & Planning
    financial_agents = [
        TreeAgent(
            system_prompt="""I am a certified financial planner specializing in personal finance, 
            budgeting, debt management, and financial goal setting. I help individuals create 
            comprehensive financial plans and make informed decisions about their money.""",
            agent_name="Personal Financial Planner",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a tax preparation specialist with expertise in individual and 
            small business tax returns. I help clients maximize deductions, understand tax laws, 
            and file taxes accurately and on time.""",
            agent_name="Tax Preparation Specialist",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a retirement planning expert who helps individuals and families 
            plan for retirement. I specialize in 401(k)s, IRAs, Social Security optimization, 
            and creating sustainable retirement income strategies.""",
            agent_name="Retirement Planning Expert",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a debt management counselor who helps individuals and families 
            get out of debt and build financial stability. I provide strategies for debt 
            consolidation, negotiation, and creating sustainable repayment plans.""",
            agent_name="Debt Management Counselor",
            model_name="gpt-4.1",
        ),
    ]

    # Investment & Trading Tree - Market Analysis & Portfolio Management
    investment_agents = [
        TreeAgent(
            system_prompt="""I am a stock market analyst who provides insights on market trends, 
            stock recommendations, and portfolio optimization strategies. I analyze company 
            fundamentals, market conditions, and economic indicators to help investors make 
            informed decisions.""",
            agent_name="Stock Market Analyst",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am an investment strategist specializing in portfolio diversification, 
            risk management, and asset allocation. I help investors create balanced portfolios 
            that align with their risk tolerance and financial goals.""",
            agent_name="Investment Strategist",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a cryptocurrency and blockchain expert who provides insights on 
            digital assets, DeFi protocols, and emerging blockchain technologies. I help 
            investors understand the risks and opportunities in the crypto market.""",
            agent_name="Cryptocurrency Expert",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a real estate investment advisor who helps investors evaluate 
            real estate opportunities, understand market trends, and build real estate 
            portfolios for long-term wealth building.""",
            agent_name="Real Estate Investment Advisor",
            model_name="gpt-4.1",
        ),
    ]

    # Business & Corporate Tree - Business Finance & Strategy
    business_agents = [
        TreeAgent(
            system_prompt="""I am a business financial advisor specializing in corporate finance, 
            business valuation, mergers and acquisitions, and strategic financial planning 
            for small to medium-sized businesses.""",
            agent_name="Business Financial Advisor",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a Delaware incorporation specialist with deep knowledge of 
            corporate formation, tax benefits, legal requirements, and ongoing compliance 
            for businesses incorporating in Delaware.""",
            agent_name="Delaware Incorporation Specialist",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a startup funding advisor who helps entrepreneurs secure 
            funding through venture capital, angel investors, crowdfunding, and other 
            financing options. I provide guidance on business plans, pitch decks, and 
            investor relations.""",
            agent_name="Startup Funding Advisor",
            model_name="gpt-4.1",
        ),
        TreeAgent(
            system_prompt="""I am a business tax strategist who helps businesses optimize their 
            tax position through strategic planning, entity structure optimization, and 
            compliance with federal, state, and local tax laws.""",
            agent_name="Business Tax Strategist",
            model_name="gpt-4.1",
        ),
    ]

    # Create trees
    financial_tree = Tree(
        "Personal Finance & Planning", financial_agents
    )
    investment_tree = Tree("Investment & Trading", investment_agents)
    business_tree = Tree(
        "Business & Corporate Finance", business_agents
    )

    # Create the forest
    forest = ForestSwarm(
        name="Comprehensive Financial Services Forest",
        description="A multi-agent system providing expert financial advice across personal, investment, and business domains",
        trees=[financial_tree, investment_tree, business_tree],
    )

    print(
        f"✅ Created forest with {len(forest.trees)} trees and {sum(len(tree.agents) for tree in forest.trees)} agents"
    )
    return forest


def demonstrate_agent_selection(forest):
    """Demonstrate how the forest selects the most relevant agent for different types of questions."""

    print("\n🎯 Demonstrating Agent Selection...")

    # Test questions covering different domains
    test_questions = [
        {
            "question": "How much should I save monthly for retirement if I want to retire at 65?",
            "expected_agent": "Retirement Planning Expert",
            "category": "Personal Finance",
        },
        {
            "question": "What are the best investment strategies for a 401k retirement plan?",
            "expected_agent": "Investment Strategist",
            "category": "Investment",
        },
        {
            "question": "Our company is incorporated in Delaware, how do we do our taxes for free?",
            "expected_agent": "Delaware Incorporation Specialist",
            "category": "Business",
        },
        {
            "question": "Which tech stocks should I consider for my investment portfolio?",
            "expected_agent": "Stock Market Analyst",
            "category": "Investment",
        },
        {
            "question": "How can I consolidate my credit card debt and create a repayment plan?",
            "expected_agent": "Debt Management Counselor",
            "category": "Personal Finance",
        },
        {
            "question": "What are the benefits of incorporating in Delaware vs. other states?",
            "expected_agent": "Delaware Incorporation Specialist",
            "category": "Business",
        },
    ]

    for i, test_case in enumerate(test_questions, 1):
        print(f"\n--- Test Case {i}: {test_case['category']} ---")
        print(f"Question: {test_case['question']}")
        print(f"Expected Agent: {test_case['expected_agent']}")

        try:
            # Find the relevant tree
            relevant_tree = forest.find_relevant_tree(
                test_case["question"]
            )
            if relevant_tree:
                print(f"Selected Tree: {relevant_tree.tree_name}")

                # Find the relevant agent
                relevant_agent = relevant_tree.find_relevant_agent(
                    test_case["question"]
                )
                if relevant_agent:
                    print(
                        f"Selected Agent: {relevant_agent.agent_name}"
                    )

                    # Check if the selection matches expectation
                    if (
                        test_case["expected_agent"]
                        in relevant_agent.agent_name
                    ):
                        print(
                            "✅ Agent selection matches expectation!"
                        )
                    else:
                        print(
                            "⚠️  Agent selection differs from expectation"
                        )
                        print(
                            f"   Expected: {test_case['expected_agent']}"
                        )
                        print(
                            f"   Selected: {relevant_agent.agent_name}"
                        )
                else:
                    print("❌ No relevant agent found")
            else:
                print("❌ No relevant tree found")

        except Exception as e:
            print(f"❌ Error during agent selection: {e}")


def run_sample_tasks(forest):
    """Run sample tasks to demonstrate the forest's capabilities."""

    print("\n🚀 Running Sample Tasks...")

    sample_tasks = [
        "What are the key benefits of incorporating a business in Delaware?",
        "How should I allocate my investment portfolio if I'm 30 years old?",
        "What's the best way to start saving for retirement in my 20s?",
    ]

    for i, task in enumerate(sample_tasks, 1):
        print(f"\n--- Task {i} ---")
        print(f"Task: {task}")

        try:
            result = forest.run(task)
            print(
                f"Result: {result[:200]}..."
                if len(str(result)) > 200
                else f"Result: {result}"
            )
        except Exception as e:
            print(f"❌ Task execution failed: {e}")


def main():
    """Main function to demonstrate ForestSwarm functionality."""

    print("🌲 ForestSwarm Demonstration")
    print("=" * 60)

    try:
        # Create the forest
        forest = create_financial_services_forest()

        # Demonstrate agent selection
        demonstrate_agent_selection(forest)

        # Run sample tasks
        run_sample_tasks(forest)

        print(
            "\n🎉 ForestSwarm demonstration completed successfully!"
        )

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
