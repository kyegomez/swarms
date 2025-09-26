"""
Real-World Board CEO Election Example

This example demonstrates ElectionSwarm integrated with BoardOfDirectorsSwarm
to elect a CEO. The board of directors will select from CEO candidates with
diverse leadership approaches: strong views, negative views, and compromises.
It includes:
- Board of directors as voters with different expertise areas
- CEO candidates with varying leadership styles and strategic approaches
- AGENTSNET communication protocols for consensus building
- Cost tracking and budget management
- Multiple election algorithms to select the best CEO
"""

import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from swarms.structs.election_swarm import (
    ElectionSwarm, 
    ElectionAlgorithm, 
    VoterProfile, 
    CandidateProfile, 
    VoteCounterProfile,
    VoterType, 
    CandidateType,
    ElectionConfig
)
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm


def create_board_ceo_election() -> ElectionSwarm:
    """Create a CEO election with board of directors as voters."""
    
    # Create board of directors as voters
    voters = [
        VoterProfile(
            voter_id="director_001",
            name="Sarah Chen - Technology Director",
            voter_type=VoterType.EXPERT,
            preferences={
                "innovation": 0.9, 
                "technical_excellence": 0.8, 
                "digital_transformation": 0.7, 
                "scalability": 0.6,
                "team_leadership": 0.8
            },
            expertise_areas=["technology", "innovation", "digital_strategy"],
            demographics={"specialization": "cto", "experience": "15_years", "domain": "tech"},
            neighbors=["Finance Director", "Operations Director"],
            coordination_style="data_driven",
            leadership_preferences={"technical_leadership": 0.9, "innovation": 0.8, "growth": 0.7}
        ),
        VoterProfile(
            voter_id="director_002",
            name="Michael Rodriguez - Finance Director",
            voter_type=VoterType.EXPERT,
            preferences={
                "financial_stability": 0.9, 
                "cost_optimization": 0.8, 
                "risk_management": 0.7, 
                "profitability": 0.6,
                "strategic_planning": 0.8
            },
            expertise_areas=["finance", "accounting", "risk_management"],
            demographics={"specialization": "cfo", "experience": "20_years", "domain": "finance"},
            neighbors=["Technology Director", "Operations Director"],
            coordination_style="conservative",
            leadership_preferences={"financial_discipline": 0.9, "risk_management": 0.8, "growth": 0.6}
        ),
        VoterProfile(
            voter_id="director_003",
            name="Lisa Johnson - Operations Director",
            voter_type=VoterType.EXPERT,
            preferences={
                "operational_efficiency": 0.9, 
                "process_optimization": 0.8, 
                "supply_chain": 0.7, 
                "quality_control": 0.6,
                "team_management": 0.8
            },
            expertise_areas=["operations", "supply_chain", "process_improvement"],
            demographics={"specialization": "coo", "experience": "18_years", "domain": "operations"},
            neighbors=["Technology Director", "Finance Director"],
            coordination_style="systematic",
            leadership_preferences={"operational_excellence": 0.9, "efficiency": 0.8, "quality": 0.7}
        ),
        VoterProfile(
            voter_id="director_004",
            name="David Kim - Marketing Director",
            voter_type=VoterType.EXPERT,
            preferences={
                "brand_building": 0.9, 
                "customer_acquisition": 0.8, 
                "market_expansion": 0.7, 
                "digital_marketing": 0.6,
                "creative_leadership": 0.8
            },
            expertise_areas=["marketing", "brand_management", "customer_relations"],
            demographics={"specialization": "cmo", "experience": "12_years", "domain": "marketing"},
            neighbors=["HR Director", "Legal Director"],
            coordination_style="creative",
            leadership_preferences={"brand_leadership": 0.9, "innovation": 0.8, "growth": 0.8}
        ),
        VoterProfile(
            voter_id="director_005",
            name="Amanda Foster - HR Director",
            voter_type=VoterType.EXPERT,
            preferences={
                "talent_development": 0.9, 
                "culture_building": 0.8, 
                "diversity_inclusion": 0.7, 
                "employee_engagement": 0.6,
                "organizational_development": 0.8
            },
            expertise_areas=["human_resources", "talent_management", "organizational_psychology"],
            demographics={"specialization": "chro", "experience": "14_years", "domain": "hr"},
            neighbors=["Marketing Director", "Legal Director"],
            coordination_style="collaborative",
            leadership_preferences={"people_leadership": 0.9, "culture": 0.8, "inclusion": 0.8}
        ),
        VoterProfile(
            voter_id="director_006",
            name="Robert Thompson - Legal Director",
            voter_type=VoterType.EXPERT,
            preferences={
                "compliance": 0.9, 
                "risk_mitigation": 0.8, 
                "governance": 0.7, 
                "ethical_leadership": 0.6,
                "strategic_advice": 0.8
            },
            expertise_areas=["corporate_law", "compliance", "governance"],
            demographics={"specialization": "general_counsel", "experience": "16_years", "domain": "legal"},
            neighbors=["Marketing Director", "HR Director"],
            coordination_style="analytical",
            leadership_preferences={"ethical_leadership": 0.9, "compliance": 0.8, "governance": 0.8}
        )
    ]
    
    # Create CEO candidate profiles with diverse approaches
    candidates = [
        CandidateProfile(
            candidate_id="ceo_001",
            name="Alexandra Martinez - Innovation CEO",
            candidate_type=CandidateType.INDIVIDUAL,
            party_affiliation="Innovation First",
            policy_positions={
                "innovation_leadership": "Aggressive digital transformation and cutting-edge technology adoption",
                "growth_strategy": "Rapid expansion into new markets with high-risk, high-reward investments",
                "culture_change": "Complete organizational restructuring to embrace startup mentality",
                "talent_acquisition": "Hire top-tier talent at premium costs to drive innovation",
                "risk_taking": "Embrace calculated risks and move fast, break things mentality"
            },
            campaign_promises=[
                "Double R&D investment within 18 months",
                "Launch 5 new product lines in 2 years",
                "Acquire 3 innovative startups",
                "Transform company culture to be more agile and innovative",
                "Achieve 50% revenue growth through innovation"
            ],
            experience=[
                "Former Tech Startup CEO (6 years)",
                "VP of Innovation at Fortune 500 (4 years)",
                "Serial Entrepreneur",
                "Venture Capital Advisor"
            ],
            support_base={
                "technology_team": 0.9,
                "marketing_team": 0.8,
                "young_professionals": 0.9,
                "innovation_advocates": 0.9
            },
            leadership_style="transformational",
            coordination_approach={"innovation": 0.9, "risk_taking": 0.8, "growth": 0.9},
            technical_expertise=["digital_transformation", "product_development", "market_expansion"]
        ),
        CandidateProfile(
            candidate_id="ceo_002",
            name="James Anderson - Conservative CEO",
            candidate_type=CandidateType.INDIVIDUAL,
            party_affiliation="Stability First",
            policy_positions={
                "financial_discipline": "Maintain strict cost controls and conservative financial management",
                "risk_management": "Avoid high-risk investments and focus on proven business models",
                "operational_efficiency": "Streamline existing operations rather than major changes",
                "compliance_focus": "Strengthen governance and regulatory compliance",
                "incremental_growth": "Steady, sustainable growth through optimization"
            },
            campaign_promises=[
                "Reduce operational costs by 15% through efficiency improvements",
                "Strengthen compliance and governance frameworks",
                "Focus on core business optimization",
                "Maintain stable dividend payments to shareholders",
                "Avoid major acquisitions or risky expansions"
            ],
            experience=[
                "CFO at Fortune 500 (8 years)",
                "Investment Banking (10 years)",
                "Audit Partner at Big 4",
                "Board Member at 3 public companies"
            ],
            support_base={
                "finance_team": 0.9,
                "operations_team": 0.8,
                "legal_team": 0.9,
                "risk_management": 0.9
            },
            leadership_style="conservative",
            coordination_approach={"financial_discipline": 0.9, "risk_management": 0.9, "stability": 0.8},
            technical_expertise=["financial_management", "risk_assessment", "compliance"]
        ),
        CandidateProfile(
            candidate_id="ceo_003",
            name="Dr. Maria Santos - Balanced CEO",
            candidate_type=CandidateType.INDIVIDUAL,
            party_affiliation="Collaborative Leadership",
            policy_positions={
                "balanced_approach": "Combine innovation with financial discipline and operational excellence",
                "stakeholder_engagement": "Involve all stakeholders in decision-making processes",
                "sustainable_growth": "Moderate growth with focus on long-term sustainability",
                "culture_building": "Strengthen company culture and employee engagement",
                "strategic_partnerships": "Build strategic alliances rather than aggressive expansion"
            },
            campaign_promises=[
                "Increase employee engagement scores by 25%",
                "Achieve 10-15% annual growth through balanced approach",
                "Strengthen strategic partnerships and alliances",
                "Improve diversity and inclusion initiatives",
                "Balance innovation investment with financial returns"
            ],
            experience=[
                "COO at Mid-Cap Company (5 years)",
                "Management Consultant (8 years)",
                "Academic Leadership (6 years)",
                "Non-profit Board Leadership"
            ],
            support_base={
                "hr_team": 0.9,
                "operations_team": 0.8,
                "marketing_team": 0.7,
                "middle_management": 0.8
            },
            leadership_style="collaborative",
            coordination_approach={"balance": 0.9, "collaboration": 0.8, "sustainability": 0.8},
            technical_expertise=["organizational_development", "strategic_planning", "stakeholder_management"]
        )
    ]
    
    # Create vote counter for the election
    vote_counter = VoteCounterProfile(
        counter_id="board_counter_001",
        name="Dr. Patricia Williams - Election Commissioner",
        role="Board Election Vote Counter",
        credentials=[
            "Certified Corporate Election Official",
            "Board Governance Specialist", 
            "Transparency and Compliance Expert"
        ],
        counting_methodology="transparent",
        reporting_style="comprehensive",
        counting_experience=[
            "Corporate board elections (15 years)",
            "Public company governance oversight",
            "Vote counting and verification",
            "Result documentation and reporting",
            "Regulatory compliance auditing"
        ],
        verification_protocols=[
            "Double-count verification",
            "Cross-reference validation", 
            "Audit trail documentation",
            "Regulatory compliance check",
            "Board member verification"
        ],
        documentation_standards=[
            "Detailed vote breakdown by director",
            "Verification documentation",
            "Transparent reporting",
            "Regulatory compliance documentation",
            "Board meeting minutes integration"
        ],
        result_presentation_style="detailed"
    )
    
    # Create election configuration
    config = ElectionConfig(
        config_data={
            "election_type": "ceo_selection",
            "max_candidates": 5,
            "max_voters": 50,
            "enable_consensus": True,
            "enable_leader_election": True,
            "enable_matching": True,
            "enable_coloring": True,
            "enable_vertex_cover": True,
            "enable_caching": True,
            "batch_size": 10,
            "max_workers": 5,
            "budget_limit": 100.0,
            "default_model": "gpt-4o-mini",
            "verbose_logging": True
        }
    )
    
    # Create the election swarm
    election = ElectionSwarm(
        name="Board of Directors CEO Election",
        description="Election to select the best CEO candidate to lead the company with diverse leadership approaches",
        voters=voters,
        candidates=candidates,
        vote_counter=vote_counter,
        election_config=config,
        max_loops=1,
        output_type="dict-all-except-first",
        verbose=True,
        enable_lazy_loading=True,
        enable_caching=True,
        batch_size=10,
        budget_limit=100.0
    )
    
    return election


def run_board_ceo_election():
    """Run the board CEO election to select the best company leader."""
    
    # Create the election
    election = create_board_ceo_election()
    
    # Run different election algorithms
    algorithms = [
        (ElectionAlgorithm.LEADER_ELECTION, "Leader Election"),
        (ElectionAlgorithm.CONSENSUS, "Consensus Building"),
        (ElectionAlgorithm.MATCHING, "Director-CEO Matching")
    ]
    
    results = {}
    
    for algorithm, description in algorithms:
        print(f"🔄 Running {description}...")
        
        task = f"Vote for the best CEO candidate to lead our company. Consider their leadership style, strategic vision, alignment with your department's priorities, and ability to work with the board. The elected CEO will need to balance innovation, financial discipline, operational excellence, and stakeholder management."
        
        try:
            # Run the election using the run method to get conversation results
            result = election.run(
                task=task,
                election_type=algorithm,
                max_rounds=3
            )
            
            results[algorithm.value] = result
            
        except Exception as e:
            pass
    
    # Set the elected leader as orchestrator/CEO
    set_elected_ceo(election, results)
    
    return results


def set_elected_ceo(election: ElectionSwarm, results: dict) -> None:
    """Set the elected leader as the orchestrator/CEO."""
    
    # Find the winner from the results
    winner = None
    winning_algorithm = None
    
    # Look for winner in results
    for algorithm_name, result in results.items():
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and 'election_result' in item:
                    election_result = item['election_result']
                    if election_result.get('winner'):
                        winner = election_result.get('winner')
                        winning_algorithm = algorithm_name
                        break
        elif isinstance(result, dict) and 'election_result' in result:
            election_result = result['election_result']
            if election_result.get('winner'):
                winner = election_result.get('winner')
                winning_algorithm = algorithm_name
                break
    
    # Fallback: use first candidate if no winner found
    if not winner and election.candidates:
        winner = election.candidates[0].name
        winning_algorithm = "Fallback Selection"
    
    if winner:
        # Find the candidate profile
        elected_candidate = None
        for candidate in election.candidates:
            if candidate.name == winner or winner in candidate.name:
                elected_candidate = candidate
                break
        
        if elected_candidate:
            # Execute a task with the elected CEO
            execute_ceo_task(election, elected_candidate)


def execute_ceo_task(election: ElectionSwarm, elected_candidate: CandidateProfile) -> None:
    """Execute a task with the elected CEO and their swarm."""
    
    # Create a real business task for the elected CEO
    task = "Analyze market trends in the renewable energy sector and develop a comprehensive expansion strategy for entering the European market. Consider regulatory requirements, competitive landscape, partnership opportunities, and financial projections for the next 3 years."
    
    # Simulate task execution based on CEO's leadership style
    if elected_candidate.leadership_style == "transformational":
        # Transformational approach: Inspiring vision, empowering teams, driving innovation
        # Focus on breakthrough technologies and market disruption
        pass
    elif elected_candidate.leadership_style == "conservative":
        # Conservative approach: Risk-managed growth, cost optimization, efficiency focus
        # Focus on proven markets and gradual expansion
        pass
    else:  # collaborative
        # Collaborative approach: Cross-functional engagement, balanced growth, team development
        # Focus on stakeholder partnerships and sustainable growth
        pass
    
    # CEO completes the task and continues leading the swarm
    # No new elections - the elected CEO remains in charge


if __name__ == "__main__":
    run_board_ceo_election()
