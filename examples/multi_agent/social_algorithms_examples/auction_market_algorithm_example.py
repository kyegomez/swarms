"""
Auction/Market-Based Social Algorithm Example

This example demonstrates a market-based social algorithm where agents bid on tasks
and an auctioneer selects the best proposals. This simulates a competitive market
environment where agents must optimize their proposals to win contracts.
"""

from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


# Create specialized agents for different aspects of the task
bidder1 = Agent(
    agent_name="Technical_Specialist",
    system_prompt="You are a technical specialist focused on innovative technical solutions. You excel at creating detailed technical proposals with specific implementation plans.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

bidder2 = Agent(
    agent_name="Business_Strategist",
    system_prompt="You are a business strategist focused on market analysis and business value. You excel at creating proposals that emphasize ROI and market potential.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

bidder3 = Agent(
    agent_name="Creative_Innovator",
    system_prompt="You are a creative innovator focused on out-of-the-box solutions and novel approaches. You excel at creating unique and creative proposals.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

auctioneer = Agent(
    agent_name="Auctioneer",
    system_prompt="You are an impartial auctioneer and evaluator. You analyze proposals based on technical merit, business value, creativity, and feasibility. You must select the best proposal and provide detailed reasoning.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

quality_assessor = Agent(
    agent_name="Quality_Assessor",
    system_prompt="You are a quality assessor focused on evaluating the feasibility and quality of proposals. You provide objective technical and practical assessments.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


def auction_market_algorithm(agents, task, **kwargs):
    """
    An auction/market-based social algorithm where agents bid on tasks
    and compete for selection based on their proposals.
    """
    bidders = agents[:-2]  # First 3 agents are bidders
    auctioneer_agent = agents[-2]  # Second to last is auctioneer
    quality_agent = agents[-1]  # Last is quality assessor

    # Phase 1: Bidding - Each agent creates a proposal
    proposals = {}
    for i, bidder in enumerate(bidders):
        bid_prompt = f"""
        Create a comprehensive proposal for this task: {task}
        
        Your proposal should include:
        1. Executive summary
        2. Technical approach
        3. Timeline and milestones
        4. Resource requirements
        5. Expected outcomes
        6. Unique value proposition
        
        Make your proposal compelling and competitive. You are bidding against other specialists.
        """

        proposal = bidder.run(bid_prompt)
        proposals[f"bidder_{i+1}_{bidder.agent_name}"] = {
            "proposal": proposal,
            "bidder_name": bidder.agent_name,
        }

    # Phase 2: Quality Assessment
    quality_assessments = {}
    for bidder_name, proposal_data in proposals.items():
        assessment_prompt = f"""
        Assess the quality and feasibility of this proposal from {proposal_data['bidder_name']}:
        
        {proposal_data['proposal']}
        
        Evaluate on:
        1. Technical feasibility (1-10)
        2. Innovation level (1-10)
        3. Resource efficiency (1-10)
        4. Timeline realism (1-10)
        5. Overall quality (1-10)
        
        Provide scores and detailed feedback for each criterion.
        """

        assessment = quality_agent.run(assessment_prompt)
        quality_assessments[bidder_name] = assessment

    # Phase 3: Auction - Auctioneer evaluates and selects winner
    auction_prompt = f"""
    As an auctioneer, evaluate these proposals and select the winner:
    
    TASK: {task}
    
    PROPOSALS:
    """

    for bidder_name, proposal_data in proposals.items():
        auction_prompt += f"\n\n{bidder_name} ({proposal_data['bidder_name']}):\n{proposal_data['proposal']}\n"
        auction_prompt += f"Quality Assessment: {quality_assessments[bidder_name]}\n"
        auction_prompt += "-" * 50

    auction_prompt += """
    
    Select the winning proposal based on:
    1. Technical excellence
    2. Business value
    3. Innovation
    4. Feasibility
    5. Overall competitiveness
    
    Provide:
    1. Winner selection
    2. Detailed reasoning
    3. Ranking of all proposals
    4. Recommendations for improvement
    """

    auction_result = auctioneer_agent.run(auction_prompt)

    # Phase 4: Contract Negotiation (simulated)
    winner_name = None
    for bidder_name, proposal_data in proposals.items():
        if proposal_data["bidder_name"] in auction_result:
            winner_name = bidder_name
            break

    if winner_name:
        winner_agent = next(
            agent
            for agent in bidders
            if agent.agent_name
            == proposals[winner_name]["bidder_name"]
        )

        negotiation_prompt = f"""
        As the winning bidder, create a detailed contract and implementation plan:
        
        Original Task: {task}
        Your Winning Proposal: {proposals[winner_name]['proposal']}
        Auctioneer Feedback: {auction_result}
        
        Create:
        1. Detailed contract terms
        2. Implementation timeline
        3. Success metrics
        4. Risk mitigation strategies
        5. Next steps
        """

        contract = winner_agent.run(negotiation_prompt)
    else:
        contract = "No clear winner identified"

    return {
        "task": task,
        "proposals": proposals,
        "quality_assessments": quality_assessments,
        "auction_result": auction_result,
        "contract": contract,
        "winner": winner_name,
        "algorithm_type": "auction_market",
    }


# Auction/Market Algorithm
social_alg = SocialAlgorithms(
    name="Auction-Market-Algorithm",
    description="Market-based algorithm where agents bid on tasks and compete for selection",
    agents=[bidder1, bidder2, bidder3, auctioneer, quality_assessor],
    social_algorithm=auction_market_algorithm,
    verbose=True,
    max_execution_time=600,  # 10 minutes for complex bidding process
)

if __name__ == "__main__":
    result = social_alg.run(
        "Develop a comprehensive AI-powered customer service platform for e-commerce"
    )

    print("=== AUCTION MARKET ALGORITHM RESULTS ===")
    print(f"Task: {result.final_outputs['task']}")
    print(f"Winner: {result.final_outputs['winner']}")
    print(f"Total Communication Steps: {result.total_steps}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print("\n=== WINNING PROPOSAL ===")
    if result.final_outputs["winner"]:
        winner_data = result.final_outputs["proposals"][
            result.final_outputs["winner"]
        ]
        print(f"Winner: {winner_data['bidder_name']}")
        print(f"Proposal: {winner_data['proposal'][:500]}...")
    print("\n=== AUCTION RESULT ===")
    print(result.final_outputs["auction_result"][:500] + "...")
