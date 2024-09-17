import os
from termcolor import colored
import json

welcome = """
Swarms is the first-ever multi-agent enterpris-grade framework that enables you to seamlessly orchestrate agents!
"""


def print_welcome():
    print(
        colored(
            f"Welcome to the Swarms Framework! \n {welcome}",
            "cyan",
            attrs=["bold"],
        )
    )
    print(
        colored(
            "Thank you for trying out Swarms. We are excited to have you on board to enable you to get started.",
            "cyan",
        )
    )
    print()
    print(colored("Resources", "cyan", attrs=["bold"]))
    print(
        colored("GitHub: ", "cyan")
        + colored("https://github.com/kyegomez/swarms", "magenta")
    )
    print(
        colored("Discord: ", "cyan")
        + colored(
            "https://discord.com/servers/agora-999382051935506503",
            "magenta",
        )
    )
    print(
        colored("Documentation: ", "cyan")
        + colored("https://docs.swarms.world", "magenta")
    )
    print(
        colored("Marketplace: ", "cyan")
        + colored("https://swarms.world", "magenta")
    )
    print(
        colored("Submit an Issue: ", "cyan")
        + colored(
            "https://github.com/kyegomez/swarms/issues/new/choose",
            "magenta",
        )
    )
    print(
        colored("Swarms Project Board // Roadmap ", "cyan")
        + colored(
            "https://github.com/users/kyegomez/projects/1", "magenta"
        )
    )
    print()
    print(
        colored(
            "Let's get to know you a bit better!",
            "magenta",
            attrs=["bold"],
        )
    )


def get_user_info():
    first_name = input(colored("What is your first name? ", "blue"))
    last_name = input(colored("What is your last name? ", "blue"))
    email = input(colored("What is your email? ", "blue"))
    company = input(
        colored("Which company do you work for? ", "blue")
    )
    project = input(
        colored("What are you trying to build with Swarms? ", "blue")
    )
    swarms_type = input(
        colored("What type of swarms are you building? ", "blue")
    )

    user_info = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "company": company,
        "project": project,
        "swarms_type": swarms_type,
    }

    return user_info


def save_user_info(user_info: dict = None):
    cache_dir = os.path.expanduser("~/.swarms_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file = os.path.join(cache_dir, "user_info.json")
    with open(cache_file, "w") as f:
        json.dump(user_info, f, indent=4)

    print(
        colored(
            "Your information has been saved as a JSON file! Thank you.",
            "cyan",
        )
    )


def onboard():
    print_welcome()
    user_info = get_user_info()
    save_user_info(user_info)
    print(
        colored(
            "You're all set! Enjoy using Swarms.",
            "cyan",
            attrs=["bold"],
        )
    )
