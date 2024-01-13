from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule


def display_markdown_message(message: str, color: str = "cyan"):
    """
    Display markdown message. Works with multiline strings with lots of indentation.
    Will automatically make single line > tags beautiful.
    """

    console = Console()
    for line in message.split("\n"):
        line = line.strip()
        if line == "":
            console.print("")
        elif line == "---":
            console.print(Rule(style=color))
        else:
            console.print(Markdown(line, style=color))

    if "\n" not in message and message.startswith(">"):
        # Aesthetic choice. For these tags, they need a space below them
        console.print("")


# display_markdown_message("I love you and you are beautiful.", "cyan")
