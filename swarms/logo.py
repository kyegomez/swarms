from rich import print as rich_print
from rich.markdown import Markdown
from rich.rule import Rule
from termcolor import colored, cprint


def display_markdown_message(message):
    """
    Display markdown message. Works with multiline strings with lots of indentation.
    Will automatically make single line > tags beautiful.
    """

    for line in message.split("\n"):
        line = line.strip()
        if line == "":
            print("")
        elif line == "---":
            rich_print(Rule(style="white"))
        else:
            rich_print(Markdown(line))

    if "\n" not in message and message.startswith(">"):
        # Aesthetic choice. For these tags, they need a space below them
        print("")


logo = """
  ________  _  _______ _______  _____   ______
 /  ___/\ \/ \/ /\__  \\_  __ \/     \ /  ___/
 \___ \  \     /  / __ \|  | \/  Y Y  \\___ \
/____  >  \/\_/  (____  /__|  |__|_|  /____  >
     \/               \/            \/     \/
"""

logo2 = """

  _________ __      __   _____  __________    _____     _________
 /   _____//  \    /  \ /  _  \ \______   \  /     \   /   _____/
 \_____  \ \   \/\/   //  /_\  \ |       _/ /  \ /  \  \_____  \
 /        \ \        //    |    \|    |   \/    Y    \ /        \
/_______  /  \__/\  / \____|__  /|____|_  /\____|__  //_______  /
        \/        \/          \/        \/         \/         \/

"""


def print_colored_logo():
    with open("swarms/logo.txt", "r") as file:
        logo = file.read()
    text = colored(logo, "red")
    print(text)


# # Call the function
# print_colored_logo()
