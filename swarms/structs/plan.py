from typing import List

from swarms.structs.step import Step


class Plan:
    def __init__(self, steps: List[Step]):
        """
        Initializes a Plan object.

        Args:
            steps (List[Step]): A list of Step objects representing the steps in the plan.
        """
        self.steps = steps

    def __str__(self) -> str:
        """
        Returns a string representation of the Plan object.

        Returns:
            str: A string representation of the Plan object.
        """
        return str([str(step) for step in self.steps])

    def __repr(self) -> str:
        """
        Returns a string representation of the Plan object.

        Returns:
            str: A string representation of the Plan object.
        """
        return str(self)
