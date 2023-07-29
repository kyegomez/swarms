"""Tool for asking human input."""


class HumanInputRun:
    """Tool that asks user for input."""

    def __init__(self, prompt_func=None, input_func=None):
        self.name = "human"
        self.description = (
            "You can ask a human for guidance when you think you "
            "got stuck or you are not sure what to do next. "
            "The input should be a question for the human."
        )
        self.prompt_func = prompt_func if prompt_func else self._print_func
        self.input_func = input_func if input_func else input

    def _print_func(self, text: str) -> None:
        print("\n")
        print(text)

    def run(self, query: str) -> str:
        """Use the Human input tool."""
        self.prompt_func(query)
        return self.input_func()