import argparse
from swarms.cli.parse_yaml import (
    create_agent_from_yaml,
    run_agent,
    list_agents,
)

SWARMS_LOGO = """
  _________                                     
 /   _____/_  _  _______ _______  _____   ______
 \_____  \\ \/ \/ /\__  \\_  __ \/     \ /  ___/
 /        \\     /  / __ \|  | \/  Y Y  \\___ \ 
/_______  / \/\_/  (____  /__|  |__|_|  /____  >
        \/              \/            \/     \/ 
"""

RED_COLOR_CODE = "\033[91m"
RESET_COLOR_CODE = "\033[0m"

# print(RED_COLOR_CODE + SWARMS_LOGO + RESET_COLOR_CODE)


def main():
    parser = argparse.ArgumentParser(
        description=f"""
        
        {SWARMS_LOGO}
        CLI for managing and running swarms agents.
        
        """
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # create agent command
    create_parser = subparsers.add_parser(
        "create", help="Create a new agent from a YAML file"
    )
    create_parser.add_argument(
        "agent", type=str, help="Path to the YAML file"
    )

    # run agent command
    run_parser = subparsers.add_parser(
        "run", help="Run an agent with a specified task"
    )
    run_parser.add_argument(
        "agent_name", type=str, help="Name of the agent to run"
    )
    run_parser.add_argument(
        "task", type=str, help="Task for the agent to execute"
    )

    # list agents command
    subparsers.add_parser("list", help="List all agents")

    # Additional help options
    parser.add_argument(
        "--issue",
        action="store_true",
        help="Open an issue on GitHub: https://github.com/kyegomez/swarms/issues/new/choose",
    )
    parser.add_argument(
        "--community",
        action="store_true",
        help="Join our community on Discord: https://discord.com/servers/agora-999382051935506503",
    )

    args = parser.parse_args()

    if args.issue:
        print(
            "Open an issue on GitHub: https://github.com/kyegomez/swarms/issues/new/choose"
        )
    elif args.community:
        print(
            "Join our community on Discord: https://discord.com/servers/agora-999382051935506503"
        )
    elif args.command == "create":
        create_agent_from_yaml(args.agent)
    elif args.command == "run":
        run_agent(args.agent_name, args.task)
    elif args.command == "list agents":
        list_agents()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
