from swarms.structs.long_agent import LongAgent


if __name__ == "__main__":
    long_agent = LongAgent(
        token_count_per_agent=3000, output_type="final"
    )
    print(long_agent.run([""]))
