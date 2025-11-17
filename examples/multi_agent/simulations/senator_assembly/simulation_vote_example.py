from swarms.sims.senator_assembly import SenatorAssembly


def main():
    senator_simulation = SenatorAssembly(
        model_name="claude-sonnet-4-20250514"
    )
    senator_simulation.simulate_vote_concurrent(
        (
            "A bill proposing a significant reduction in federal income tax rates for all American citizens. "
            "The legislation aims to lower tax brackets across the board, increase the standard deduction, "
            "and provide additional tax relief for middle- and lower-income families. Proponents argue that "
            "the bill will stimulate economic growth, increase disposable income, and enhance consumer spending. "
            "Opponents raise concerns about the potential impact on the federal deficit, funding for public services, "
            "and long-term fiscal responsibility. Senators must weigh the economic, social, and budgetary implications "
            "before casting their votes."
        ),
        batch_size=10,
    )


if __name__ == "__main__":
    main()
