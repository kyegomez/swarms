"""
EuroSwarm Parliament - Example Script

This script demonstrates the comprehensive democratic functionality of the EuroSwarm Parliament,
including bill introduction, committee work, parliamentary debates, and democratic voting.
"""

import json
import time
from datetime import datetime

# Import directly from the file
from euroswarm_parliament import (
    EuroSwarmParliament,
    VoteType,
    ParliamentaryRole,
    ParliamentaryMember
)


def demonstrate_parliament_initialization():
    """Demonstrate parliament initialization and basic functionality with cost optimization."""
    
    print("\nEUROSWARM PARLIAMENT INITIALIZATION DEMONSTRATION (COST OPTIMIZED)")
    print("=" * 60)
    
    # Initialize the parliament with cost optimization
    parliament = EuroSwarmParliament(
        eu_data_file="EU.xml",
        parliament_size=None,  # Use all MEPs from EU.xml (717)
        enable_democratic_discussion=True,
        enable_committee_work=True,
        enable_amendment_process=True,
        enable_lazy_loading=True,  # NEW: Lazy load MEP agents
        enable_caching=True,  # NEW: Enable response caching
        batch_size=25,  # NEW: Batch size for concurrent execution
        budget_limit=100.0,  # NEW: Budget limit in dollars
        verbose=True
    )
    
    print(f"Parliament initialized with {len(parliament.meps)} MEPs")
    
    # Show parliament composition with cost stats
    composition = parliament.get_parliament_composition()
    
    print(f"\nPARLIAMENT COMPOSITION:")
    print(f"Total MEPs: {composition['total_meps']}")
    print(f"Loaded MEPs: {composition['loaded_meps']} (lazy loading active)")
    
    print(f"\nCOST OPTIMIZATION:")
    cost_stats = composition['cost_stats']
    print(f"Budget Limit: ${cost_stats['budget_remaining'] + cost_stats['total_cost']:.2f}")
    print(f"Budget Used: ${cost_stats['total_cost']:.2f}")
    print(f"Budget Remaining: ${cost_stats['budget_remaining']:.2f}")
    print(f"Cache Hit Rate: {cost_stats['cache_hit_rate']:.1%}")
    
    print(f"\nPOLITICAL GROUP DISTRIBUTION:")
    for group, data in composition['political_groups'].items():
        count = data['count']
        percentage = data['percentage']
        print(f"  {group}: {count} MEPs ({percentage:.1f}%)")
    
    print(f"\nCOMMITTEE LEADERSHIP:")
    for committee_name, committee_data in composition['committees'].items():
        chair = committee_data['chair']
        if chair:
            print(f"  {committee_name}: {chair}")
    
    return parliament


def demonstrate_individual_mep_interaction(parliament):
    """Demonstrate individual MEP interaction and personality."""
    
    print("\nINDIVIDUAL MEP INTERACTION DEMONSTRATION")
    print("=" * 60)
    
    # Get a sample MEP
    sample_mep_name = list(parliament.meps.keys())[0]
    sample_mep = parliament.meps[sample_mep_name]
    
    print(f"Sample MEP: {sample_mep.full_name}")
    print(f"Country: {sample_mep.country}")
    print(f"Political Group: {sample_mep.political_group}")
    print(f"National Party: {sample_mep.national_party}")
    print(f"Committees: {', '.join(sample_mep.committees)}")
    print(f"Expertise Areas: {', '.join(sample_mep.expertise_areas)}")
    
    # Test MEP agent interaction
    if sample_mep.agent:
        test_prompt = "What are your views on European integration and how do you approach cross-border cooperation?"
        
        print(f"\nMEP Response to: '{test_prompt}'")
        print("-" * 50)
        
        try:
            response = sample_mep.agent.run(test_prompt)
            print(response[:500] + "..." if len(response) > 500 else response)
        except Exception as e:
            print(f"Error getting MEP response: {e}")


def demonstrate_committee_work(parliament):
    """Demonstrate committee work and hearings."""
    
    print("\nCOMMITTEE WORK DEMONSTRATION")
    print("=" * 60)
    
    # Get a real MEP as sponsor
    sponsor = list(parliament.meps.keys())[0]
    
    # Create a test bill
    bill = parliament.introduce_bill(
        title="European Digital Rights and Privacy Protection Act",
        description="Comprehensive legislation to strengthen digital rights, enhance privacy protection, and establish clear guidelines for data handling across the European Union.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Legal Affairs",
        sponsor=sponsor
    )
    
    print(f"Bill: {bill.title}")
    print(f"Committee: {bill.committee}")
    print(f"Sponsor: {bill.sponsor}")
    
    # Conduct committee hearing
    print(f"\nCONDUCTING COMMITTEE HEARING...")
    hearing_result = parliament.conduct_committee_hearing(bill.committee, bill)
    
    print(f"Committee: {hearing_result['committee']}")
    print(f"Participants: {len(hearing_result['participants'])} MEPs")
    print(f"Recommendation: {hearing_result['recommendations']['recommendation']}")
    print(f"Support: {hearing_result['recommendations']['support_percentage']:.1f}%")
    print(f"Oppose: {hearing_result['recommendations']['oppose_percentage']:.1f}%")
    print(f"Amend: {hearing_result['recommendations']['amend_percentage']:.1f}%")


def demonstrate_parliamentary_debate(parliament):
    """Demonstrate parliamentary debate functionality."""
    
    print("\nPARLIAMENTARY DEBATE DEMONSTRATION")
    print("=" * 60)
    
    # Get a real MEP as sponsor
    sponsor = list(parliament.meps.keys())[1]
    
    # Create a test bill
    bill = parliament.introduce_bill(
        title="European Green Deal Implementation Act",
        description="Legislation to implement the European Green Deal, including carbon neutrality targets, renewable energy investments, and sustainable development measures.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Environment, Public Health and Food Safety",
        sponsor=sponsor
    )
    
    print(f"Bill: {bill.title}")
    print(f"Description: {bill.description}")
    
    # Conduct parliamentary debate
    print(f"\nCONDUCTING PARLIAMENTARY DEBATE...")
    debate_result = parliament.conduct_parliamentary_debate(bill, max_speakers=10)
    
    print(f"Debate Participants: {len(debate_result['participants'])} MEPs")
    print(f"Debate Analysis:")
    print(f"  Support: {debate_result['analysis']['support_count']} speakers ({debate_result['analysis']['support_percentage']:.1f}%)")
    print(f"  Oppose: {debate_result['analysis']['oppose_count']} speakers ({debate_result['analysis']['oppose_percentage']:.1f}%)")
    print(f"  Neutral: {debate_result['analysis']['neutral_count']} speakers ({debate_result['analysis']['neutral_percentage']:.1f}%)")


def demonstrate_democratic_voting(parliament):
    """Demonstrate democratic voting functionality."""
    
    print("\nDEMOCRATIC VOTING DEMONSTRATION")
    print("=" * 60)
    
    # Get a real MEP as sponsor
    sponsor = list(parliament.meps.keys())[2]
    
    # Create a test bill
    bill = parliament.introduce_bill(
        title="European Social Rights and Labor Protection Act",
        description="Legislation to strengthen social rights, improve labor conditions, and ensure fair treatment of workers across the European Union.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Employment and Social Affairs",
        sponsor=sponsor
    )
    
    print(f"Bill: {bill.title}")
    print(f"Sponsor: {bill.sponsor}")
    
    # Conduct democratic vote
    print(f"\nCONDUCTING DEMOCRATIC VOTE...")
    vote_result = parliament.conduct_democratic_vote(bill)
    
    # Calculate percentages
    total_votes = vote_result.votes_for + vote_result.votes_against + vote_result.abstentions
    in_favor_percentage = (vote_result.votes_for / total_votes * 100) if total_votes > 0 else 0
    against_percentage = (vote_result.votes_against / total_votes * 100) if total_votes > 0 else 0
    abstentions_percentage = (vote_result.abstentions / total_votes * 100) if total_votes > 0 else 0
    
    print(f"Vote Results:")
    print(f"  Total Votes: {total_votes}")
    print(f"  In Favor: {vote_result.votes_for} ({in_favor_percentage:.1f}%)")
    print(f"  Against: {vote_result.votes_against} ({against_percentage:.1f}%)")
    print(f"  Abstentions: {vote_result.abstentions} ({abstentions_percentage:.1f}%)")
    print(f"  Result: {vote_result.result.value}")
    
    # Show political group breakdown if available
    if hasattr(vote_result, 'group_votes') and vote_result.group_votes:
        print(f"\nPOLITICAL GROUP BREAKDOWN:")
        for group, votes in vote_result.group_votes.items():
            print(f"  {group}: {votes['in_favor']}/{votes['total']} in favor ({votes['percentage']:.1f}%)")
    else:
        print(f"\nIndividual votes recorded: {len(vote_result.individual_votes)} MEPs")


def demonstrate_complete_democratic_session(parliament):
    """Demonstrate a complete democratic parliamentary session."""
    
    print("\nCOMPLETE DEMOCRATIC SESSION DEMONSTRATION")
    print("=" * 60)
    
    # Get a real MEP as sponsor
    sponsor = list(parliament.meps.keys())[3]
    
    # Run complete session
    session_result = parliament.run_democratic_session(
        bill_title="European Innovation and Technology Advancement Act",
        bill_description="Comprehensive legislation to promote innovation, support technology startups, and establish Europe as a global leader in digital transformation and technological advancement.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Industry, Research and Energy",
        sponsor=sponsor
    )
    
    print(f"Session Results:")
    print(f"  Bill: {session_result['bill'].title}")
    print(f"  Committee Hearing: {session_result['hearing']['recommendations']['recommendation']}")
    print(f"  Debate Participants: {len(session_result['debate']['participants'])} MEPs")
    print(f"  Final Vote: {session_result['vote']['result']}")
    print(f"  Vote Margin: {session_result['vote']['in_favor_percentage']:.1f}% in favor")


def demonstrate_political_analysis(parliament):
    """Demonstrate political analysis and voting prediction."""
    
    print("\nPOLITICAL ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Get a real MEP as sponsor
    sponsor = list(parliament.meps.keys())[4]
    
    # Create a test bill
    bill = parliament.introduce_bill(
        title="European Climate Action and Sustainability Act",
        description="Comprehensive climate action legislation including carbon pricing, renewable energy targets, and sustainable development measures.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Environment, Public Health and Food Safety",
        sponsor=sponsor
    )
    
    print(f"Bill: {bill.title}")
    print(f"Sponsor: {bill.sponsor}")
    
    # Analyze political landscape
    analysis = parliament.analyze_political_landscape(bill)
    
    print(f"\nPOLITICAL LANDSCAPE ANALYSIS:")
    print(f"  Overall Support: {analysis['overall_support']:.1f}%")
    print(f"  Opposition: {analysis['opposition']:.1f}%")
    print(f"  Uncertainty: {analysis['uncertainty']:.1f}%")
    
    print(f"\nPOLITICAL GROUP ANALYSIS:")
    for group, data in analysis['group_analysis'].items():
        print(f"  {group}: {data['support']:.1f}% support, {data['opposition']:.1f}% opposition")


def demonstrate_hierarchical_democratic_voting(parliament):
    """Demonstrate hierarchical democratic voting with political group boards."""
    
    print("\nHIERARCHICAL DEMOCRATIC VOTING DEMONSTRATION")
    print("=" * 60)
    
    # Get a real MEP as sponsor
    sponsor = list(parliament.meps.keys())[5]
    
    # Create a test bill
    bill = parliament.introduce_bill(
        title="European Climate Action and Sustainability Act",
        description="Comprehensive climate action legislation including carbon pricing, renewable energy targets, and sustainable development measures.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Environment, Public Health and Food Safety",
        sponsor=sponsor
    )
    
    print(f"Bill: {bill.title}")
    print(f"Sponsor: {bill.sponsor}")
    
    # Conduct hierarchical vote
    print(f"\nCONDUCTING HIERARCHICAL DEMOCRATIC VOTE...")
    hierarchical_result = parliament.conduct_hierarchical_democratic_vote(bill)
    
    print(f"Hierarchical Vote Results:")
    print(f"  Total Votes: {hierarchical_result['total_votes']}")
    print(f"  In Favor: {hierarchical_result['in_favor']} ({hierarchical_result['in_favor_percentage']:.1f}%)")
    print(f"  Against: {hierarchical_result['against']} ({hierarchical_result['against_percentage']:.1f}%)")
    print(f"  Result: {hierarchical_result['result']}")
    
    print(f"\nPOLITICAL GROUP BOARD DECISIONS:")
    for group, decision in hierarchical_result['group_decisions'].items():
        print(f"  {group}: {decision['decision']} ({decision['confidence']:.1f}% confidence)")


def demonstrate_complete_hierarchical_session(parliament):
    """Demonstrate a complete hierarchical democratic session."""
    
    print("\nCOMPLETE HIERARCHICAL DEMOCRATIC SESSION DEMONSTRATION")
    print("=" * 60)
    
    # Get a real MEP as sponsor
    sponsor = list(parliament.meps.keys())[6]
    
    # Run complete hierarchical session
    session_result = parliament.run_hierarchical_democratic_session(
        bill_title="European Climate Action and Sustainability Act",
        bill_description="Comprehensive climate action legislation including carbon pricing, renewable energy targets, and sustainable development measures.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Environment, Public Health and Food Safety",
        sponsor=sponsor
    )
    
    print(f"Hierarchical Session Results:")
    print(f"  Bill: {session_result['bill'].title}")
    print(f"  Committee Hearing: {session_result['hearing']['recommendations']['recommendation']}")
    print(f"  Debate Participants: {len(session_result['debate']['participants'])} MEPs")
    print(f"  Final Vote: {session_result['vote']['result']}")
    print(f"  Vote Margin: {session_result['vote']['in_favor_percentage']:.1f}% in favor")


def demonstrate_wikipedia_personalities(parliament):
    """Demonstrate the Wikipedia personality system for realistic MEP behavior."""
    
    print("\nWIKIPEDIA PERSONALITY SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Check if Wikipedia personalities are available
    if not parliament.enable_wikipedia_personalities:
        print("Wikipedia personality system not available")
        print("To enable: Install required dependencies and run Wikipedia scraper")
        return
    
    print(f"Wikipedia personality system enabled")
    print(f"Loaded {len(parliament.personality_profiles)} personality profiles")
    
    # Show sample personality profiles
    print(f"\nSAMPLE PERSONALITY PROFILES:")
    print("-" * 40)
    
    sample_count = 0
    for mep_name, profile in parliament.personality_profiles.items():
        if sample_count >= 3:  # Show only 3 samples
            break
            
        print(f"\n{mep_name}")
        print(f"   Wikipedia URL: {profile.wikipedia_url if profile.wikipedia_url else 'Not available'}")
        print(f"   Summary: {profile.summary[:200]}..." if profile.summary else "No summary available")
        print(f"   Political Views: {profile.political_views[:150]}..." if profile.political_views else "Based on party alignment")
        print(f"   Policy Focus: {profile.policy_focus[:150]}..." if profile.policy_focus else "General parliamentary work")
        print(f"   Achievements: {profile.achievements[:150]}..." if profile.achievements else "Parliamentary service")
        print(f"   Last Updated: {profile.last_updated}")
        
        sample_count += 1
    
    # Demonstrate personality-driven voting
    print(f"\nPERSONALITY-DRIVEN VOTING DEMONSTRATION:")
    print("-" * 50)
    
    # Create a test bill that would trigger different personality responses
    bill = parliament.introduce_bill(
        title="European Climate Action and Green Technology Investment Act",
        description="Comprehensive legislation to accelerate Europe's transition to renewable energy, including massive investments in green technology, carbon pricing mechanisms, and support for affected industries and workers.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Environment",
        sponsor="Climate Action Leader"
    )
    
    print(f"Bill: {bill.title}")
    print(f"Description: {bill.description}")
    
    # Show how different MEPs with Wikipedia personalities would respond
    print(f"\nPERSONALITY-BASED RESPONSES:")
    print("-" * 40)
    
    sample_meps = list(parliament.personality_profiles.keys())[:3]
    
    for mep_name in sample_meps:
        mep = parliament.meps.get(mep_name)
        profile = parliament.personality_profiles.get(mep_name)
        
        if mep and profile:
            print(f"\n{mep_name} ({mep.political_group})")
            
            # Show personality influence
            if profile.political_views:
                print(f"   Political Views: {profile.political_views[:100]}...")
            
            if profile.policy_focus:
                print(f"   Policy Focus: {profile.policy_focus[:100]}...")
            
            # Predict voting behavior based on personality
            if "environment" in profile.policy_focus.lower() or "climate" in profile.political_views.lower():
                predicted_vote = "LIKELY SUPPORT"
                reasoning = "Environmental policy focus and climate advocacy"
            elif "economic" in profile.policy_focus.lower() or "business" in profile.political_views.lower():
                predicted_vote = "LIKELY OPPOSE"
                reasoning = "Economic concerns about investment costs"
            else:
                predicted_vote = "UNCERTAIN"
                reasoning = "Mixed considerations based on party alignment"
            
            print(f"   Predicted Vote: {predicted_vote}")
            print(f"   Reasoning: {reasoning}")
    
    # Demonstrate scraping functionality
    print(f"\nWIKIPEDIA SCRAPING CAPABILITIES:")
    print("-" * 50)
    print("Can scrape Wikipedia data for all 717 MEPs")
    print("Extracts political views, career history, and achievements")
    print("Creates detailed personality profiles in JSON format")
    print("Integrates real personality data into AI agent system prompts")
    print("Enables realistic, personality-driven voting behavior")
    print("Respectful API usage with configurable delays")
    
    print(f"\nTo scrape all MEP personalities:")
    print("   parliament.scrape_wikipedia_personalities(delay=1.0)")
    print("   # This will create personality profiles for all 717 MEPs")
    print("   # Profiles are saved in 'mep_personalities/' directory")


def demonstrate_optimized_parliamentary_session(parliament):
    """Demonstrate cost-optimized parliamentary session."""
    
    print("\nCOST-OPTIMIZED PARLIAMENTARY SESSION DEMONSTRATION")
    print("=" * 60)
    
    # Run optimized session with cost limit
    session_result = parliament.run_optimized_parliamentary_session(
        bill_title="European Digital Rights and Privacy Protection Act",
        bill_description="Comprehensive legislation to strengthen digital rights, enhance privacy protection, and establish clear guidelines for data handling across the European Union.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Legal Affairs",
        max_cost=25.0  # Max $25 for this session
    )
    
    print(f"Session Results:")
    print(f"  Bill: {session_result['session_summary']['bill_title']}")
    print(f"  Final Outcome: {session_result['session_summary']['final_outcome']}")
    print(f"  Total Cost: ${session_result['session_summary']['total_cost']:.2f}")
    print(f"  Budget Remaining: ${session_result['cost_stats']['budget_remaining']:.2f}")
    
    # Show detailed cost statistics
    cost_stats = parliament.get_cost_statistics()
    print(f"\nDETAILED COST STATISTICS:")
    print(f"  Total Tokens Used: {cost_stats['total_tokens']:,}")
    print(f"  Requests Made: {cost_stats['requests_made']}")
    print(f"  Cache Hits: {cost_stats['cache_hits']}")
    print(f"  Cache Hit Rate: {cost_stats['cache_hit_rate']:.1%}")
    print(f"  Loading Efficiency: {cost_stats['loading_efficiency']:.1%}")
    print(f"  Cache Size: {cost_stats['cache_size']} entries")
    
    return session_result


def main():
    """Main demonstration function."""
    
    print("EUROSWARM PARLIAMENT - COST OPTIMIZED DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows the EuroSwarm Parliament with cost optimization features:")
    print("• Lazy loading of MEP agents (only create when needed)")
    print("• Response caching (avoid repeated API calls)")
    print("• Batch processing (control memory and cost)")
    print("• Budget controls (hard limits on spending)")
    print("• Cost tracking (real-time monitoring)")
    
    # Initialize parliament with cost optimization
    parliament = demonstrate_parliament_initialization()
    
    # Demonstrate individual MEP interaction (will trigger lazy loading)
    demonstrate_individual_mep_interaction(parliament)
    
    # Demonstrate committee work with cost optimization
    demonstrate_committee_work(parliament)
    
    # Demonstrate parliamentary debate with cost optimization
    demonstrate_parliamentary_debate(parliament)
    
    # Demonstrate democratic voting with cost optimization
    demonstrate_democratic_voting(parliament)
    
    # Demonstrate political analysis with cost optimization
    demonstrate_political_analysis(parliament)
    
    # Demonstrate optimized parliamentary session
    demonstrate_optimized_parliamentary_session(parliament)
    
    # Show final cost statistics
    final_stats = parliament.get_cost_statistics()
    print(f"\nFINAL COST STATISTICS:")
    print(f"Total Cost: ${final_stats['total_cost']:.2f}")
    print(f"Budget Remaining: ${final_stats['budget_remaining']:.2f}")
    print(f"Cache Hit Rate: {final_stats['cache_hit_rate']:.1%}")
    print(f"Loading Efficiency: {final_stats['loading_efficiency']:.1%}")
    
    print(f"\n✅ COST OPTIMIZATION DEMONSTRATION COMPLETED!")
    print(f"✅ EuroSwarm Parliament now supports cost-effective large-scale simulations")
    print(f"✅ Lazy loading: {final_stats['loaded_meps']}/{final_stats['total_meps']} MEPs loaded")
    print(f"✅ Caching: {final_stats['cache_hit_rate']:.1%} hit rate")
    print(f"✅ Budget control: ${final_stats['total_cost']:.2f} spent of ${final_stats['budget_remaining'] + final_stats['total_cost']:.2f} budget")


if __name__ == "__main__":
    main() 