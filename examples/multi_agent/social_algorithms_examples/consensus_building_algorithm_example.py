from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


# Create agents representing different stakeholder perspectives
stakeholder1 = Agent(
    agent_name="Technical_Lead",
    system_prompt="You represent the technical team's perspective. You focus on technical feasibility, implementation complexity, and technical risks. You advocate for technically sound solutions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

stakeholder2 = Agent(
    agent_name="Business_Manager",
    system_prompt="You represent the business perspective. You focus on ROI, market impact, customer value, and business strategy. You advocate for solutions that drive business value.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

stakeholder3 = Agent(
    agent_name="User_Advocate",
    system_prompt="You represent the end-user perspective. You focus on user experience, usability, accessibility, and user needs. You advocate for user-centric solutions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

stakeholder4 = Agent(
    agent_name="Security_Expert",
    system_prompt="You represent the security and compliance perspective. You focus on security risks, compliance requirements, and risk mitigation. You advocate for secure and compliant solutions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

facilitator = Agent(
    agent_name="Facilitator",
    system_prompt="You are a neutral facilitator focused on managing the consensus building process. You help identify common ground, mediate conflicts, and guide the group toward agreement. You remain neutral and objective.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

voting_coordinator = Agent(
    agent_name="Voting_Coordinator",
    system_prompt="You are a voting coordinator focused on managing the voting process, tallying votes, and ensuring fair and transparent decision-making. You handle the mechanics of voting and consensus measurement.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


def consensus_building_algorithm(agents, task, **kwargs):
    """
    A consensus building algorithm where agents work together to reach agreement
    through voting, negotiation, and compromise.
    """
    stakeholders = agents[:-2]  # First 4 agents are stakeholders
    facilitator_agent = agents[-2]  # Second to last is facilitator
    voting_agent = agents[-1]  # Last is voting coordinator

    max_rounds = kwargs.get("max_rounds", 5)
    consensus_threshold = kwargs.get(
        "consensus_threshold", 0.75
    )  # 75% agreement required

    # Phase 1: Initial Position Statements
    facilitator_agent.run(
        f"Starting consensus building process for: {task}"
    )

    initial_positions = {}
    for i, stakeholder in enumerate(stakeholders):
        position_prompt = f"""
        As {stakeholder.agent_name}, provide your initial position on: {task}
        
        Include:
        1. Your core requirements and priorities
        2. Your main concerns and constraints
        3. Your preferred approach or solution
        4. What you're willing to compromise on
        5. What you absolutely cannot accept
        
        Be clear and specific about your position.
        """

        position = stakeholder.run(position_prompt)
        initial_positions[stakeholder.agent_name] = position

    # Phase 2: Iterative Consensus Building Rounds
    consensus_rounds = []
    current_consensus_level = 0.0

    for round_num in range(1, max_rounds + 1):
        facilitator_agent.run(f"Starting consensus round {round_num}")

        # Round Discussion - Each stakeholder responds to others' positions
        round_discussion = {}
        for stakeholder in stakeholders:
            # Create context from other stakeholders' positions
            other_positions = "\n".join(
                [
                    f"{name}: {pos}"
                    for name, pos in initial_positions.items()
                    if name != stakeholder.agent_name
                ]
            )

            discussion_prompt = f"""
            Round {round_num} Discussion:
            
            Your original position: {initial_positions[stakeholder.agent_name]}
            
            Other stakeholders' positions:
            {other_positions}
            
            Respond to:
            1. What do you agree with from other positions?
            2. What concerns do you have about other positions?
            3. How might you modify your position to find common ground?
            4. What compromises are you willing to make?
            5. What new ideas or solutions emerge from this discussion?
            """

            discussion = stakeholder.run(discussion_prompt)
            round_discussion[stakeholder.agent_name] = discussion

        # Facilitator identifies common ground and areas of disagreement
        facilitation_prompt = f"""
        As facilitator, analyze this round {round_num} discussion:
        
        {round_discussion}
        
        Identify:
        1. Areas of agreement and common ground
        2. Key areas of disagreement
        3. Potential compromise solutions
        4. Next steps for reaching consensus
        5. Specific recommendations for each stakeholder
        """

        facilitation_guidance = facilitator_agent.run(
            facilitation_prompt
        )

        # Voting on current proposals
        voting_prompt = f"""
        As voting coordinator, conduct a vote on the current state of consensus:
        
        Round {round_num} Discussion: {round_discussion}
        Facilitator Guidance: {facilitation_guidance}
        
        Create a voting mechanism where each stakeholder votes on:
        1. Overall agreement level (1-10)
        2. Willingness to proceed with current direction (Yes/No)
        3. Confidence in reaching consensus (1-10)
        
        Calculate consensus metrics and determine if threshold is met.
        """

        voting_result = voting_agent.run(voting_prompt)

        # Extract consensus level (simplified heuristic)
        consensus_level = 0.5  # Default
        if "agreement level" in voting_result.lower():
            try:
                import re

                scores = re.findall(r"(\d+(?:\.\d+)?)", voting_result)
                if scores:
                    consensus_level = (
                        float(scores[0]) / 10.0
                    )  # Normalize to 0-1
            except:
                pass

        current_consensus_level = consensus_level

        consensus_rounds.append(
            {
                "round": round_num,
                "discussion": round_discussion,
                "facilitation": facilitation_guidance,
                "voting": voting_result,
                "consensus_level": consensus_level,
            }
        )

        # Check if consensus is reached
        if consensus_level >= consensus_threshold:
            facilitator_agent.run(
                f"Consensus reached at {consensus_level:.2f} (threshold: {consensus_threshold})"
            )
            break

        # Update positions based on discussion
        for stakeholder in stakeholders:
            update_prompt = f"""
            Based on round {round_num} discussion and facilitator guidance, 
            update your position:
            
            Your previous position: {initial_positions[stakeholder.agent_name]}
            Round discussion: {round_discussion[stakeholder.agent_name]}
            Facilitator guidance: {facilitation_guidance}
            
            Provide your updated position for the next round.
            """

            updated_position = stakeholder.run(update_prompt)
            initial_positions[stakeholder.agent_name] = (
                updated_position
            )

    # Phase 3: Final Consensus Document
    if current_consensus_level >= consensus_threshold:
        consensus_prompt = f"""
        Create a final consensus document based on the agreement reached:
        
        Task: {task}
        Final Positions: {initial_positions}
        Consensus Level: {current_consensus_level}
        Rounds Completed: {len(consensus_rounds)}
        
        Include:
        1. Agreed-upon solution/approach
        2. Key compromises made
        3. Implementation plan
        4. Success metrics
        5. Next steps
        """

        final_consensus = facilitator_agent.run(consensus_prompt)
    else:
        final_consensus = f"""
        Consensus not reached after {max_rounds} rounds.
        Final consensus level: {current_consensus_level}
        Threshold required: {consensus_threshold}
        
        Areas of continued disagreement:
        {initial_positions}
        
        Recommendations for next steps.
        """

    # Phase 4: Final Validation
    validation_prompt = f"""
    As voting coordinator, provide final validation of the consensus process:
    
    Final Consensus: {final_consensus}
    Process Summary: {len(consensus_rounds)} rounds completed
    Final Consensus Level: {current_consensus_level}
    
    Validate the process and provide final recommendations.
    """

    final_validation = voting_agent.run(validation_prompt)

    return {
        "task": task,
        "initial_positions": initial_positions,
        "consensus_rounds": consensus_rounds,
        "final_consensus_level": current_consensus_level,
        "consensus_threshold": consensus_threshold,
        "consensus_reached": current_consensus_level
        >= consensus_threshold,
        "final_consensus": final_consensus,
        "final_validation": final_validation,
        "total_rounds": len(consensus_rounds),
        "algorithm_type": "consensus_building",
    }


# Consensus Building Algorithm
social_alg = SocialAlgorithms(
    name="Consensus-Building-Algorithm",
    description="Consensus building algorithm with voting, negotiation, and compromise",
    agents=[
        stakeholder1,
        stakeholder2,
        stakeholder3,
        stakeholder4,
        facilitator,
        voting_coordinator,
    ],
    social_algorithm=consensus_building_algorithm,
    verbose=True,
    max_execution_time=1200,  # 20 minutes for multiple consensus rounds
)

if __name__ == "__main__":
    result = social_alg.run(
        "Decide on the technology stack for a new enterprise application",
        algorithm_args={"max_rounds": 5, "consensus_threshold": 0.75},
    )

    print("=== CONSENSUS BUILDING ALGORITHM RESULTS ===")
    print(f"Task: {result.final_outputs['task']}")
    print(
        f"Consensus Reached: {result.final_outputs['consensus_reached']}"
    )
    print(
        f"Final Consensus Level: {result.final_outputs['final_consensus_level']:.2f}"
    )
    print(f"Total Rounds: {result.final_outputs['total_rounds']}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Total Communication Steps: {result.total_steps}")

    print("\n=== INITIAL POSITIONS ===")
    for stakeholder, position in result.final_outputs[
        "initial_positions"
    ].items():
        print(f"{stakeholder}: {position[:200]}...")
        print()

    print("\n=== CONSENSUS ROUNDS ===")
    for round_data in result.final_outputs["consensus_rounds"]:
        print(
            f"Round {round_data['round']} - Consensus Level: {round_data['consensus_level']:.2f}"
        )
        print(f"Facilitation: {round_data['facilitation'][:200]}...")
        print()

    print("\n=== FINAL CONSENSUS ===")
    print(result.final_outputs["final_consensus"][:500] + "...")
