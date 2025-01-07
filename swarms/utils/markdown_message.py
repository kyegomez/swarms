from swarms.utils.formatter import formatter


def display_markdown_message(message: str, color: str = "cyan"):
    """
    Display markdown message. Works with multiline strings with lots of indentation.
    Will automatically make single line > tags beautiful.
    """

    for line in message.split("\n"):
        line = line.strip()
        if line == "":
            print()
        elif line == "---":
            formatter.print_panel("-" * 50)
        else:
            formatter.print_panel(line)

    if "\n" not in message and message.startswith(">"):
        # Aesthetic choice. For these tags, they need a space below them
        print()
