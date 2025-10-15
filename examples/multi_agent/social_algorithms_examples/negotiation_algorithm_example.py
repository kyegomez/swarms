from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


# Create agents representing different negotiating parties
party1 = Agent(
    agent_name="Company_Representative",
    system_prompt="You represent a company in negotiations. You focus on maximizing business value, protecting company interests, and achieving favorable terms. You are strategic and business-oriented.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

party2 = Agent(
    agent_name="Client_Representative",
    system_prompt="You represent a client in negotiations. You focus on getting the best value, ensuring quality, and protecting client interests. You are value-focused and quality-conscious.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

party3 = Agent(
    agent_name="Technical_Expert",
    system_prompt="You are a technical expert in negotiations. You focus on technical feasibility, implementation details, and technical risks. You provide objective technical assessments.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

mediator = Agent(
    agent_name="Mediator",
    system_prompt="You are a neutral mediator in negotiations. You help facilitate communication, identify common ground, suggest compromises, and guide parties toward agreement. You remain impartial and objective.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

legal_advisor = Agent(
    agent_name="Legal_Advisor",
    system_prompt="You are a legal advisor in negotiations. You focus on legal compliance, risk assessment, contract terms, and legal implications. You ensure all agreements are legally sound.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


def negotiation_algorithm(agents, task, **kwargs):
    """
    A negotiation algorithm where agents engage in back-and-forth communication
    to reach mutually acceptable agreements.
    """
    negotiating_parties = agents[
        :-2
    ]  # First 3 agents are negotiating parties
    mediator_agent = agents[-2]  # Second to last is mediator
    legal_agent = agents[-1]  # Last is legal advisor

    max_rounds = kwargs.get("max_rounds", 8)
    agreement_threshold = kwargs.get(
        "agreement_threshold", 0.8
    )  # 80% agreement required

    # Initialize negotiation state
    negotiation_history = []
    current_positions = {}
    negotiation_topics = []
    agreement_levels = []

    # Phase 1: Initial Position Statements
    mediator_agent.run(f"Starting negotiation process for: {task}")

    # Each party states their initial position
    for i, party in enumerate(negotiating_parties):
        position_prompt = f"""
        As {party.agent_name}, state your initial position for: {task}
        
        Include:
        1. Your primary objectives and goals
        2. Your key requirements and constraints
        3. Your preferred terms and conditions
        4. What you're willing to negotiate on
        5. What you absolutely cannot compromise on
        6. Your ideal outcome
        
        Be clear and specific about your position.
        """

        initial_position = party.run(position_prompt)
        current_positions[party.agent_name] = initial_position

        negotiation_history.append(
            {
                "round": 0,
                "phase": "initial_positions",
                "party": party.agent_name,
                "position": initial_position,
            }
        )

    # Phase 2: Negotiation Rounds
    for round_num in range(1, max_rounds + 1):
        mediator_agent.run(f"Starting negotiation round {round_num}")

        # Mediator analyzes current positions and suggests approach
        mediation_prompt = f"""
        As mediator, analyze the current negotiation state:
        
        Round: {round_num}
        Current Positions: {current_positions}
        Previous Rounds: {negotiation_history[-3:] if len(negotiation_history) >= 3 else negotiation_history}
        
        Provide guidance on:
        1. Areas of potential agreement
        2. Key sticking points and conflicts
        3. Suggested compromise approaches
        4. Topics to focus on this round
        5. Negotiation strategy recommendations
        """

        mediation_guidance = mediator_agent.run(mediation_prompt)

        # Each party responds to others' positions and makes counter-proposals
        round_responses = {}
        for party in negotiating_parties:
            # Create context from other parties' positions
            other_positions = "\n".join(
                [
                    f"{name}: {pos}"
                    for name, pos in current_positions.items()
                    if name != party.agent_name
                ]
            )

            response_prompt = f"""
            Round {round_num} Response:
            
            Your current position: {current_positions[party.agent_name]}
            Other parties' positions: {other_positions}
            Mediator guidance: {mediation_guidance}
            
            Respond with:
            1. What you agree with from other positions
            2. What you disagree with and why
            3. Your counter-proposals and alternatives
            4. What compromises you're willing to make
            5. What you need from other parties
            6. Your updated position
            """

            response = party.run(response_prompt)
            round_responses[party.agent_name] = response

            # Update current position
            current_positions[party.agent_name] = response

        # Legal advisor reviews proposals for legal implications
        legal_review_prompt = f"""
        As legal advisor, review the current negotiation proposals:
        
        Round: {round_num}
        Current Positions: {current_positions}
        Mediator Guidance: {mediation_guidance}
        
        Assess:
        1. Legal compliance and implications
        2. Potential legal risks and issues
        3. Contract terms and conditions
        4. Legal recommendations and warnings
        5. Suggested legal safeguards
        """

        legal_review = legal_agent.run(legal_review_prompt)

        # Record round results
        negotiation_history.append(
            {
                "round": round_num,
                "phase": "negotiation",
                "mediation_guidance": mediation_guidance,
                "responses": round_responses,
                "legal_review": legal_review,
            }
        )

        # Assess agreement level
        agreement_prompt = f"""
        Assess the current level of agreement:
        
        Round: {round_num}
        Current Positions: {current_positions}
        
        Rate agreement level (0-1) on:
        1. Overall alignment of positions
        2. Willingness to compromise
        3. Clarity of requirements
        4. Feasibility of solutions
        5. Legal compliance
        
        Provide overall agreement score and specific areas of progress.
        """

        agreement_assessment = mediator_agent.run(agreement_prompt)

        # Extract agreement level (simplified)
        agreement_level = 0.5  # Default
        if "agreement score" in agreement_assessment.lower():
            try:
                import re

                scores = re.findall(
                    r"(\d+(?:\.\d+)?)", agreement_assessment
                )
                if scores:
                    agreement_level = float(scores[0])
            except:
                pass

        agreement_levels.append(agreement_level)

        # Check if agreement is reached
        if agreement_level >= agreement_threshold:
            mediator_agent.run(
                f"Agreement reached at level {agreement_level:.2f} (threshold: {agreement_threshold})"
            )
            break

        # Mediator provides feedback for next round
        feedback_prompt = f"""
        Provide feedback for the next round:
        
        Round: {round_num}
        Agreement Level: {agreement_level}
        Current Positions: {current_positions}
        
        Suggest:
        1. What worked well this round
        2. What needs improvement
        3. Focus areas for next round
        4. Specific negotiation tactics
        5. Compromise suggestions
        """

        round_feedback = mediator_agent.run(feedback_prompt)
        negotiation_history.append(
            {
                "round": round_num,
                "phase": "feedback",
                "agreement_level": agreement_level,
                "feedback": round_feedback,
            }
        )

    # Phase 3: Final Agreement or Impasse
    if (
        agreement_levels
        and agreement_levels[-1] >= agreement_threshold
    ):
        # Create final agreement
        agreement_prompt = f"""
        Create the final agreement based on the negotiation:
        
        Task: {task}
        Final Positions: {current_positions}
        Agreement Level: {agreement_levels[-1]}
        Negotiation History: {len(negotiation_history)} rounds
        
        Include:
        1. Agreed-upon terms and conditions
        2. Key compromises made
        3. Implementation timeline
        4. Success metrics and KPIs
        5. Dispute resolution procedures
        6. Next steps and responsibilities
        """

        final_agreement = mediator_agent.run(agreement_prompt)

        # Legal review of final agreement
        final_legal_review = legal_agent.run(
            f"""
        Perform final legal review of the agreement:
        
        {final_agreement}
        
        Ensure legal compliance and provide final legal approval.
        """
        )

        negotiation_result = {
            "status": "agreement_reached",
            "agreement": final_agreement,
            "legal_review": final_legal_review,
            "agreement_level": agreement_levels[-1],
        }
    else:
        # Negotiation impasse
        impasse_prompt = f"""
        Address the negotiation impasse:
        
        Task: {task}
        Final Positions: {current_positions}
        Final Agreement Level: {agreement_levels[-1] if agreement_levels else 0}
        Threshold Required: {agreement_threshold}
        
        Provide:
        1. Analysis of why agreement wasn't reached
        2. Key sticking points
        3. Recommendations for resolving impasse
        4. Alternative approaches
        5. Next steps
        """

        impasse_analysis = mediator_agent.run(impasse_prompt)

        negotiation_result = {
            "status": "impasse",
            "analysis": impasse_analysis,
            "final_agreement_level": (
                agreement_levels[-1] if agreement_levels else 0
            ),
            "sticking_points": current_positions,
        }

    # Final negotiation summary
    summary_prompt = f"""
    Provide a comprehensive summary of the negotiation process:
    
    Task: {task}
    Total Rounds: {len(negotiation_history)}
    Final Status: {negotiation_result['status']}
    Agreement Levels: {agreement_levels}
    
    Summarize the negotiation process and outcomes.
    """

    negotiation_summary = mediator_agent.run(summary_prompt)

    return {
        "task": task,
        "negotiation_history": negotiation_history,
        "current_positions": current_positions,
        "agreement_levels": agreement_levels,
        "negotiation_result": negotiation_result,
        "negotiation_summary": negotiation_summary,
        "total_rounds": len(negotiation_history),
        "algorithm_type": "negotiation",
    }


# Negotiation Algorithm
social_alg = SocialAlgorithms(
    name="Negotiation-Algorithm",
    description="Negotiation algorithm with back-and-forth communication and mediation",
    agents=[party1, party2, party3, mediator, legal_advisor],
    social_algorithm=negotiation_algorithm,
    verbose=True,
    max_execution_time=1800,  # 30 minutes for complex negotiations
)

if __name__ == "__main__":
    result = social_alg.run(
        "Negotiate a software development contract with specific requirements and timeline",
        algorithm_args={"max_rounds": 8, "agreement_threshold": 0.8},
    )

    print("=== NEGOTIATION ALGORITHM RESULTS ===")
    print(f"Task: {result.final_outputs['task']}")
    print(f"Total Rounds: {result.final_outputs['total_rounds']}")
    print(
        f"Final Status: {result.final_outputs['negotiation_result']['status']}"
    )
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Total Communication Steps: {result.total_steps}")

    print("\n=== AGREEMENT LEVELS ===")
    for i, level in enumerate(
        result.final_outputs["agreement_levels"]
    ):
        print(f"Round {i+1}: {level:.2f}")

    print("\n=== FINAL POSITIONS ===")
    for party, position in result.final_outputs[
        "current_positions"
    ].items():
        print(f"{party}: {position[:200]}...")
        print()

    print("\n=== NEGOTIATION ROUNDS ===")
    for round_data in result.final_outputs["negotiation_history"]:
        if round_data["phase"] == "negotiation":
            print(f"Round {round_data['round']}:")
            print(
                f"  Mediation: {round_data['mediation_guidance'][:100]}..."
            )
            print(
                f"  Legal Review: {round_data['legal_review'][:100]}..."
            )
            print()

    print("\n=== FINAL RESULT ===")
    if (
        result.final_outputs["negotiation_result"]["status"]
        == "agreement_reached"
    ):
        print("AGREEMENT REACHED:")
        print(
            result.final_outputs["negotiation_result"]["agreement"][
                :500
            ]
            + "..."
        )
    else:
        print("NEGOTIATION IMPASSE:")
        print(
            result.final_outputs["negotiation_result"]["analysis"][
                :500
            ]
            + "..."
        )

    print("\n=== NEGOTIATION SUMMARY ===")
    print(result.final_outputs["negotiation_summary"][:500] + "...")
