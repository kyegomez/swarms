from swarms.models.base_llm import AbstractLLM


try:
    import multion

except ImportError:
    raise ImportError(
        "Cannot import multion, please install 'pip install'"
    )


class MultiOn(AbstractLLM):
    """
    MultiOn is a wrapper for the Multion API.

    Args:
        **kwargs:

    Methods:
        run(self, task: str, url: str, *args, **kwargs)

    Example:
    >>> from swarms.models.multion import MultiOn
    >>> multion = MultiOn()
    >>> multion.run("Order chicken tendies", "https://www.google.com/")
    "Order chicken tendies. https://www.google.com/"

    """

    def __init__(self, **kwargs):
        super(MultiOn, self).__init__(**kwargs)

    def run(self, task: str, url: str, *args, **kwargs) -> str:
        """Run the multion model

        Args:
            task (str): _description_
            url (str): _description_

        Returns:
            str: _description_
        """
        response = multion.new_session({"input": task, "url": url})
        return response

    def generate_summary(
        self, task: str, url: str, *args, **kwargs
    ) -> str:
        """Generate a summary from the multion model

        Args:
            task (str): _description_
            url (str): _description_

        Returns:
            str: _description_
        """
        response = multion.new_session({"input": task, "url": url})
        return response
