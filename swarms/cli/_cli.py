import argparse

from swarms.agents.simple_agent import SimpleAgent, get_llm_by_name


def main():
    parser = argparse.ArgumentParser(
        prog="swarms",
        description=(
            "Run the SimpleAgent with a specified language model."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Run the SimpleAgent."
    )
    run_parser.add_argument(
        "modelname",
        type=str,
        help="The name of the language model to use.",
    )
    run_parser.add_argument(
        "--iters",
        type=int,
        default="automatic",
        help=(
            'Number of iterations or "automatic" for infinite loop.'
            ' Defaults to "automatic".'
        ),
    )

    # Add a help command
    help_parser = subparsers.add_parser(
        "help", help="Show this help message and exit."
    )
    help_parser.set_defaults(func=lambda args: parser.print_help())

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    elif args.command == "run":
        llm = get_llm_by_name(args.modelname)
        if llm is None:
            raise ValueError(
                "No language model found with name"
                f" '{args.modelname}'"
            )
        SimpleAgent(llm, iters=args.iters)


# if __name__ == "__main__":
#     main()
