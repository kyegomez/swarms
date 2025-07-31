"""
US Senate Simulation - Comprehensive Example Script

This script demonstrates various scenarios and use cases for the senator simulation,
including debates, votes, committee hearings, and individual senator interactions.
"""

from simulations.senator_assembly.senator_simulation import SenatorSimulation
import json
import time

def demonstrate_individual_senators():
    """Demonstrate individual senator responses and characteristics."""
    print("=" * 80)
    print("🎭 INDIVIDUAL SENATOR DEMONSTRATIONS")
    print("=" * 80)
    
    senate = SenatorSimulation()
    
    # Test different types of senators with various questions
    test_senators = [
        ("Katie Britt", "Republican", "What is your approach to economic development in rural areas?"),
        ("Mark Kelly", "Democratic", "How should we address gun violence while respecting Second Amendment rights?"),
        ("Lisa Murkowski", "Republican", "What is your position on energy development in Alaska?"),
        ("Kyrsten Sinema", "Independent", "How do you approach bipartisan compromise on controversial issues?"),
        ("Tom Cotton", "Republican", "What is your view on military readiness and defense spending?"),
        ("Alex Padilla", "Democratic", "How should we reform the immigration system?"),
        ("Michael Bennet", "Democratic", "What is your approach to education reform?"),
        ("Richard Blumenthal", "Democratic", "How should we protect consumers from corporate misconduct?")
    ]
    
    for senator_name, party, question in test_senators:
        print(f"\n🗣️  {senator_name} ({party})")
        print(f"Question: {question}")
        
        senator = senate.get_senator(senator_name)
        if senator:
            try:
                response = senator.run(question)
                print(f"Response: {response[:300]}...")
            except Exception as e:
                print(f"Error: {e}")
        
        print("-" * 60)

def demonstrate_senate_debates():
    """Demonstrate Senate debates on various topics."""
    print("\n" + "=" * 80)
    print("💬 SENATE DEBATE SIMULATIONS")
    print("=" * 80)
    
    senate = SenatorSimulation()
    
    debate_topics = [
        {
            "topic": "Climate change legislation and carbon pricing",
            "participants": ["Katie Britt", "Mark Kelly", "Lisa Murkowski", "Alex Padilla", "John Hickenlooper"],
            "description": "Debate on comprehensive climate change legislation"
        },
        {
            "topic": "Infrastructure spending and funding mechanisms",
            "participants": ["Kyrsten Sinema", "Tom Cotton", "Michael Bennet", "Tom Carper", "Chris Coons"],
            "description": "Debate on infrastructure investment and how to pay for it"
        },
        {
            "topic": "Healthcare reform and the Affordable Care Act",
            "participants": ["Richard Blumenthal", "Chris Murphy", "John Boozman", "Laphonza Butler", "Dan Sullivan"],
            "description": "Debate on healthcare policy and reform"
        }
    ]
    
    for i, debate_config in enumerate(debate_topics, 1):
        print(f"\n🎤 DEBATE #{i}: {debate_config['description']}")
        print(f"Topic: {debate_config['topic']}")
        print(f"Participants: {', '.join(debate_config['participants'])}")
        print("-" * 60)
        
        try:
            debate = senate.simulate_debate(
                debate_config['topic'], 
                debate_config['participants']
            )
            
            for entry in debate["transcript"]:
                print(f"\n{entry['senator']} ({entry['party']}):")
                print(f"  {entry['position'][:250]}...")
                print()
        
        except Exception as e:
            print(f"Error in debate simulation: {e}")
        
        print("=" * 60)

def demonstrate_senate_votes():
    """Demonstrate Senate voting on various bills."""
    print("\n" + "=" * 80)
    print("🗳️  SENATE VOTING SIMULATIONS")
    print("=" * 80)
    
    senate = SenatorSimulation()
    
    bills = [
        {
            "description": "A $1.2 trillion infrastructure bill including roads, bridges, broadband, and clean energy projects",
            "participants": ["Katie Britt", "Mark Kelly", "Lisa Murkowski", "Alex Padilla", "Tom Cotton", "Michael Bennet"],
            "name": "Infrastructure Investment and Jobs Act"
        },
        {
            "description": "A bill to expand background checks for gun purchases and implement red flag laws",
            "participants": ["Richard Blumenthal", "Chris Murphy", "Tom Cotton", "Mark Kelly", "Kyrsten Sinema"],
            "name": "Gun Safety and Background Check Expansion Act"
        },
        {
            "description": "A bill to provide a pathway to citizenship for DACA recipients and other undocumented immigrants",
            "participants": ["Alex Padilla", "Kyrsten Sinema", "Michael Bennet", "Tom Cotton", "Lisa Murkowski"],
            "name": "Dream Act and Immigration Reform"
        },
        {
            "description": "A bill to increase defense spending by 5% and modernize military equipment",
            "participants": ["Tom Cotton", "Dan Sullivan", "Mark Kelly", "Richard Blumenthal", "John Boozman"],
            "name": "National Defense Authorization Act"
        }
    ]
    
    for i, bill in enumerate(bills, 1):
        print(f"\n📋 VOTE #{i}: {bill['name']}")
        print(f"Bill: {bill['description']}")
        print(f"Voting Senators: {', '.join(bill['participants'])}")
        print("-" * 60)
        
        try:
            vote = senate.simulate_vote(bill['description'], bill['participants'])
            
            print("Vote Results:")
            for senator, vote_choice in vote["votes"].items():
                party = senate._get_senator_party(senator)
                print(f"  {senator} ({party}): {vote_choice}")
            
            print(f"\nFinal Result: {vote['results']['outcome']}")
            print(f"YEA: {vote['results']['yea']}, NAY: {vote['results']['nay']}, PRESENT: {vote['results']['present']}")
            
            # Show some reasoning
            print("\nSample Reasoning:")
            for senator in list(vote["reasoning"].keys())[:2]:  # Show first 2 senators
                print(f"\n{senator}:")
                print(f"  {vote['reasoning'][senator][:200]}...")
        
        except Exception as e:
            print(f"Error in vote simulation: {e}")
        
        print("=" * 60)

def demonstrate_committee_hearings():
    """Demonstrate Senate committee hearings."""
    print("\n" + "=" * 80)
    print("🏛️  COMMITTEE HEARING SIMULATIONS")
    print("=" * 80)
    
    senate = SenatorSimulation()
    
    hearings = [
        {
            "committee": "Armed Services",
            "topic": "Military readiness and defense spending priorities",
            "witnesses": ["Secretary of Defense", "Chairman of the Joint Chiefs of Staff", "Defense Industry Representative"],
            "description": "Armed Services Committee hearing on military readiness"
        },
        {
            "committee": "Environment and Public Works",
            "topic": "Climate change and environmental protection measures",
            "witnesses": ["EPA Administrator", "Climate Scientist", "Energy Industry Representative"],
            "description": "Environment Committee hearing on climate action"
        },
        {
            "committee": "Health, Education, Labor, and Pensions",
            "topic": "Healthcare access and affordability",
            "witnesses": ["HHS Secretary", "Healthcare Provider", "Patient Advocate"],
            "description": "HELP Committee hearing on healthcare"
        }
    ]
    
    for i, hearing_config in enumerate(hearings, 1):
        print(f"\n🎤 HEARING #{i}: {hearing_config['description']}")
        print(f"Committee: {hearing_config['committee']}")
        print(f"Topic: {hearing_config['topic']}")
        print(f"Witnesses: {', '.join(hearing_config['witnesses'])}")
        print("-" * 60)
        
        try:
            hearing = senate.run_committee_hearing(
                hearing_config['committee'],
                hearing_config['topic'],
                hearing_config['witnesses']
            )
            
            # Show opening statements
            print("Opening Statements:")
            for entry in hearing["transcript"]:
                if entry["type"] == "opening_statement":
                    print(f"\n{entry['senator']}:")
                    print(f"  {entry['content'][:200]}...")
            
            # Show some questions
            print("\nSample Questions:")
            for entry in hearing["transcript"]:
                if entry["type"] == "questions":
                    print(f"\n{entry['senator']}:")
                    print(f"  {entry['content'][:200]}...")
                    break  # Just show first senator's questions
        
        except Exception as e:
            print(f"Error in committee hearing: {e}")
        
        print("=" * 60)

def demonstrate_party_analysis():
    """Demonstrate party-based analysis and comparisons."""
    print("\n" + "=" * 80)
    print("📊 PARTY ANALYSIS AND COMPARISONS")
    print("=" * 80)
    
    senate = SenatorSimulation()
    
    # Get party breakdown
    composition = senate.get_senate_composition()
    print(f"Senate Composition:")
    print(json.dumps(composition, indent=2))
    
    # Compare party positions on key issues
    key_issues = [
        "Tax policy and economic stimulus",
        "Healthcare reform and the role of government",
        "Climate change and environmental regulation",
        "Immigration policy and border security"
    ]
    
    for issue in key_issues:
        print(f"\n🎯 Issue: {issue}")
        print("-" * 40)
        
        # Get Republican perspective
        republicans = senate.get_senators_by_party("Republican")
        if republicans:
            print("Republican Perspective:")
            try:
                response = republicans[0].run(f"What is the Republican position on: {issue}")
                print(f"  {response[:200]}...")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Get Democratic perspective
        democrats = senate.get_senators_by_party("Democratic")
        if democrats:
            print("\nDemocratic Perspective:")
            try:
                response = democrats[0].run(f"What is the Democratic position on: {issue}")
                print(f"  {response[:200]}...")
            except Exception as e:
                print(f"  Error: {e}")
        
        print()

def demonstrate_interactive_scenarios():
    """Demonstrate interactive scenarios and what-if situations."""
    print("\n" + "=" * 80)
    print("🎮 INTERACTIVE SCENARIOS")
    print("=" * 80)
    
    senate = SenatorSimulation()
    
    scenarios = [
        {
            "title": "Supreme Court Nomination",
            "description": "Simulate a Supreme Court nomination vote",
            "action": lambda: senate.simulate_vote(
                "Confirmation of a Supreme Court nominee with moderate judicial philosophy",
                ["Kyrsten Sinema", "Lisa Murkowski", "Mark Kelly", "Tom Cotton", "Alex Padilla"]
            )
        },
        {
            "title": "Budget Reconciliation",
            "description": "Simulate a budget reconciliation vote (simple majority)",
            "action": lambda: senate.simulate_vote(
                "Budget reconciliation bill including healthcare, climate, and tax provisions",
                ["Katie Britt", "Mark Kelly", "Michael Bennet", "Tom Cotton", "Richard Blumenthal"]
            )
        },
        {
            "title": "Bipartisan Infrastructure Deal",
            "description": "Simulate a bipartisan infrastructure agreement",
            "action": lambda: senate.simulate_debate(
                "Bipartisan infrastructure deal with traditional funding mechanisms",
                ["Kyrsten Sinema", "Lisa Murkowski", "Mark Kelly", "Tom Carper", "Chris Coons"]
            )
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 Scenario #{i}: {scenario['title']}")
        print(f"Description: {scenario['description']}")
        print("-" * 60)
        
        try:
            result = scenario['action']()
            
            if 'votes' in result:  # Vote result
                print("Vote Results:")
                for senator, vote in result['votes'].items():
                    print(f"  {senator}: {vote}")
                print(f"Outcome: {result['results']['outcome']}")
            
            elif 'transcript' in result:  # Debate result
                print("Debate Positions:")
                for entry in result['transcript'][:3]:  # Show first 3
                    print(f"\n{entry['senator']} ({entry['party']}):")
                    print(f"  {entry['position'][:150]}...")
        
        except Exception as e:
            print(f"Error in scenario: {e}")
        
        print("=" * 60)

def main():
    """Run all demonstration scenarios."""
    print("🏛️  US SENATE SIMULATION - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases various aspects of the Senate simulation")
    print("including individual senator responses, debates, votes, and committee hearings.")
    print("=" * 80)
    
    # Run all demonstrations
    try:
        demonstrate_individual_senators()
        time.sleep(2)
        
        demonstrate_senate_debates()
        time.sleep(2)
        
        demonstrate_senate_votes()
        time.sleep(2)
        
        demonstrate_committee_hearings()
        time.sleep(2)
        
        demonstrate_party_analysis()
        time.sleep(2)
        
        demonstrate_interactive_scenarios()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")
    
    print("\n" + "=" * 80)
    print("✅ SENATE SIMULATION DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("The simulation successfully demonstrated:")
    print("• Individual senator characteristics and responses")
    print("• Senate debates on various topics")
    print("• Voting simulations on different bills")
    print("• Committee hearing scenarios")
    print("• Party-based analysis and comparisons")
    print("• Interactive scenarios and what-if situations")
    print("\nYou can now use the SenatorSimulation class to create your own scenarios!")

if __name__ == "__main__":
    main() 