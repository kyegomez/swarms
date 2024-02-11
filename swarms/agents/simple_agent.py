from swarms.structs.conversation import Conversation
from swarms.models.base_llm import AbstractLLM
from typing import Any
import importlib
import pkgutil
import swarms.models


def get_llm_by_name(name: str):
    """
    Searches all the modules exported from the 'swarms.models' path for a class with the given name.

    Args:
        name (str): The name of the class to search for.

    Returns:
        type: The class with the given name, or None if no such class is found.
    """
    for importer, modname, ispkg in pkgutil.iter_modules(
        swarms.models.__path__
    ):
        module = importlib.import_module(f"swarms.models.{modname}")
        if hasattr(module, name):
            return getattr(module, name)
    return None


# Run the language model in a loop for n iterations
def SimpleAgent(
    llm: AbstractLLM = None, iters: Any = "automatic", *args, **kwargs
):
    """
    A simple agent that interacts with a language model.

    Args:
        llm (AbstractLLM): The language model to use for generating responses.
        iters (Any): The number of iterations or "automatic" to run indefinitely.
        *args: Additional positional arguments to pass to the language model.
        **kwargs: Additional keyword arguments to pass to the language model.

    Raises:
        Exception: If the language model is not defined or cannot be found.

    Returns:
        None
    """
    try:
        if llm is None:
            raise Exception("Language model not defined")

        if isinstance(llm, str):
            llm = get_llm_by_name(llm)
            if llm is None:
                raise Exception(f"Language model {llm} not found")
            llm = llm(*args, **kwargs)
    except Exception as error:
        print(f"[ERROR][SimpleAgent] {error}")
        raise error

    try:
        conv = Conversation(*args, **kwargs)
        if iters == "automatic":
            i = 0
            while True:
                user_input = input("\033[91mUser:\033[0m ")
                conv.add("user", user_input)
                if user_input.lower() == "quit":
                    break
                task = (
                    conv.return_history_as_string()
                )  # Get the conversation history
                out = llm(task, *args, **kwargs)
                conv.add("assistant", out)
                print(
                    f"\033[94mAssistant:\033[0m {out}",
                )
                conv.display_conversation()
                conv.export_conversation("conversation.txt")
                i += 1
        else:
            for i in range(iters):
                user_input = input("\033[91mUser:\033[0m ")
                conv.add("user", user_input)
                if user_input.lower() == "quit":
                    break
                task = (
                    conv.return_history_as_string()
                )  # Get the conversation history
                out = llm(task, *args, **kwargs)
                conv.add("assistant", out)
                print(
                    f"\033[94mAssistant:\033[0m {out}",
                )
                conv.display_conversation()
                conv.export_conversation("conversation.txt")

    except Exception as error:
        print(f"[ERROR][SimpleAgentConversation] {error}")
        raise error

    except KeyboardInterrupt:
        print("[INFO][SimpleAgentConversation] Keyboard interrupt")
        conv.export_conversation("conversation.txt")
        raise KeyboardInterrupt
