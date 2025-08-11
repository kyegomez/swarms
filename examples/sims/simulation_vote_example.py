from swarms.sims.senator_assembly import SenatorAssembly


def main():
    """
    Simulate a Senate vote on a bill to invade Cuba and claim it as the 51st state.

    This function initializes the SenatorAssembly and runs a concurrent vote simulation
    on the specified bill.
    """
    senator_simulation = SenatorAssembly()
    senator_simulation.simulate_vote_concurrent(
        "A bill proposing to deregulate the IPO (Initial Public Offering) market in the United States as extensively as possible. The bill seeks to remove or significantly reduce existing regulatory requirements and oversight for companies seeking to go public, with the aim of increasing market efficiency and access to capital. Senators must consider the potential economic, legal, and ethical consequences of such broad deregulation, and cast their votes accordingly.",
        batch_size=10,
    )


if __name__ == "__main__":
    main()
