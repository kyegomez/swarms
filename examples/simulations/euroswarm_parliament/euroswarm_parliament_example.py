"""
EuroSwarm Parliament - Simple Example

A basic demonstration of the EuroSwarm Parliament functionality.
"""

from euroswarm_parliament import EuroSwarmParliament, VoteType


def main():
    """Simple demonstration of EuroSwarm Parliament."""

    print("EUROSWARM PARLIAMENT - SIMPLE EXAMPLE")
    print("=" * 50)

    # Initialize the parliament
    parliament = EuroSwarmParliament(
        eu_data_file="EU.xml",
        enable_democratic_discussion=True,
        enable_committee_work=True,
        verbose=True,
    )

    print(f"Parliament initialized with {len(parliament.meps)} MEPs")

    # Get a sample MEP
    sample_mep_name = list(parliament.meps.keys())[0]
    sample_mep = parliament.meps[sample_mep_name]

    print(f"\nSample MEP: {sample_mep.full_name}")
    print(f"Country: {sample_mep.country}")
    print(f"Political Group: {sample_mep.political_group}")

    # Create a simple bill
    bill = parliament.introduce_bill(
        title="European Digital Rights Act",
        description="Basic legislation to protect digital rights across the EU.",
        bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee="Legal Affairs",
        sponsor=sample_mep_name,
    )

    print(f"\nBill introduced: {bill.title}")
    print(f"Committee: {bill.committee}")

    # Conduct a simple vote
    print("\nConducting democratic vote...")
    vote_result = parliament.conduct_democratic_vote(bill)

    print("Vote Results:")
    print(f"  In Favor: {vote_result.votes_for}")
    print(f"  Against: {vote_result.votes_against}")
    print(f"  Abstentions: {vote_result.abstentions}")
    print(f"  Result: {vote_result.result.value}")

    print("\n✅ Simple example completed!")


if __name__ == "__main__":
    main()
