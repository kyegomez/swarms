#!/usr/bin/env python3
"""
Test script for the new concurrent voting functionality in the Senate simulation.
"""

from swarms.sims.senator_assembly import SenatorAssembly


def test_concurrent_voting():
    """
    Test the new concurrent voting functionality.
    """
    print("🏛️  Testing Concurrent Senate Voting...")

    # Create the simulation
    senate = SenatorAssembly()

    print("\n📊 Senate Composition:")
    composition = senate.get_senate_composition()
    print(f"   Total Senators: {composition['total_senators']}")
    print(f"   Party Breakdown: {composition['party_breakdown']}")

    # Test concurrent voting on a bill
    bill_description = "A comprehensive infrastructure bill including roads, bridges, broadband expansion, and clean energy projects with a total cost of $1.2 trillion"

    print("\n🗳️  Running Concurrent Vote on Infrastructure Bill")
    print(f"   Bill: {bill_description[:100]}...")

    # Run the concurrent vote with batch size of 10
    vote_results = senate.simulate_vote_concurrent(
        bill_description=bill_description,
        batch_size=10,  # Process 10 senators concurrently in each batch
    )

    # Display results
    print("\n📊 Final Vote Results:")
    print(f"   Total Votes: {vote_results['results']['total_votes']}")
    print(f"   YEA: {vote_results['results']['yea']}")
    print(f"   NAY: {vote_results['results']['nay']}")
    print(f"   PRESENT: {vote_results['results']['present']}")
    print(f"   OUTCOME: {vote_results['results']['outcome']}")

    print("\n📈 Party Breakdown:")
    for party, votes in vote_results["party_breakdown"].items():
        total_party_votes = sum(votes.values())
        if total_party_votes > 0:
            print(
                f"   {party}: YEA={votes['yea']}, NAY={votes['nay']}, PRESENT={votes['present']}"
            )

    print("\n📋 Sample Individual Votes (first 10):")
    for i, (senator, vote) in enumerate(
        vote_results["votes"].items()
    ):
        if i >= 10:  # Only show first 10
            break
        party = senate._get_senator_party(senator)
        print(f"   {senator} ({party}): {vote}")

    if len(vote_results["votes"]) > 10:
        print(
            f"   ... and {len(vote_results['votes']) - 10} more votes"
        )

    print("\n⚡ Performance Info:")
    print(f"   Batch Size: {vote_results['batch_size']}")
    print(f"   Total Batches: {vote_results['total_batches']}")

    return vote_results


def test_concurrent_voting_with_subset():
    """
    Test concurrent voting with a subset of senators.
    """
    print("\n" + "=" * 60)
    print("🏛️  Testing Concurrent Voting with Subset of Senators...")

    # Create the simulation
    senate = SenatorAssembly()

    # Select a subset of senators for testing
    test_senators = [
        "Katie Britt",
        "Mark Kelly",
        "Lisa Murkowski",
        "Alex Padilla",
        "Tom Cotton",
        "Kyrsten Sinema",
        "John Barrasso",
        "Tammy Duckworth",
        "Ted Cruz",
        "Amy Klobuchar",
    ]

    bill_description = (
        "A bill to increase the federal minimum wage to $15 per hour"
    )

    print("\n🗳️  Running Concurrent Vote on Minimum Wage Bill")
    print(f"   Bill: {bill_description}")
    print(f"   Participants: {len(test_senators)} senators")

    # Run the concurrent vote
    vote_results = senate.simulate_vote_concurrent(
        bill_description=bill_description,
        participants=test_senators,
        batch_size=5,  # Smaller batch size for testing
    )

    # Display results
    print("\n📊 Vote Results:")
    print(f"   YEA: {vote_results['results']['yea']}")
    print(f"   NAY: {vote_results['results']['nay']}")
    print(f"   PRESENT: {vote_results['results']['present']}")
    print(f"   OUTCOME: {vote_results['results']['outcome']}")

    print("\n📋 All Individual Votes:")
    for senator, vote in vote_results["votes"].items():
        party = senate._get_senator_party(senator)
        print(f"   {senator} ({party}): {vote}")

    return vote_results


if __name__ == "__main__":
    # Test full senate concurrent voting
    full_results = test_concurrent_voting()

    # Test subset concurrent voting
    subset_results = test_concurrent_voting_with_subset()

    print("\n✅ Concurrent voting tests completed successfully!")
    print(f"   Full Senate: {full_results['results']['outcome']}")
    print(f"   Subset: {subset_results['results']['outcome']}")
