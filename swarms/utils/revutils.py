from functools import wraps
import logging
import os
import re
import secrets

import time
from prompt_toolkit import prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from swarms.schemas.typings import Colors

bindings = KeyBindings()

# BASE_URL = environ.get("CHATGPT_BASE_URL", "http://192.168.250.249:9898/api/")
BASE_URL = os.environ.get("CHATGPT_BASE_URL", "https://ai.fakeopen.com/api/")
# BASE_URL = environ.get("CHATGPT_BASE_URL", "https://bypass.churchless.tech/")


def create_keybindings(key: str = "c-@") -> KeyBindings:
    """
    Create keybindings for prompt_toolkit. Default key is ctrl+space.
    For possible keybindings, see: https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/key_bindings.html#list-of-special-keys
    """

    @bindings.add(key)
    def _(event: dict) -> None:
        event.app.exit(result=event.app.current_buffer.text)

    return bindings


def create_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


def create_completer(commands: list, pattern_str: str = "$") -> WordCompleter:
    return WordCompleter(words=commands, pattern=re.compile(pattern_str))


def get_input(
    session: PromptSession = None,
    completer: WordCompleter = None,
    key_bindings: KeyBindings = None,
) -> str:
    """
    Multiline input function.
    """
    return (session.prompt(
        completer=completer,
        multiline=True,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=key_bindings,
    ) if session else prompt(multiline=True))


async def get_input_async(
    session: PromptSession = None,
    completer: WordCompleter = None,
) -> str:
    """
    Multiline input function.
    """
    return (await session.prompt_async(
        completer=completer,
        multiline=True,
        auto_suggest=AutoSuggestFromHistory(),
    ) if session else prompt(multiline=True))


def get_filtered_keys_from_object(obj: object, *keys: str) -> any:
    """
    Get filtered list of object variable names.
    :param keys: List of keys to include. If the first key is "not", the remaining keys will be removed from the class keys.
    :return: List of class keys.
    """
    class_keys = obj.__dict__.keys()
    if not keys:
        return set(class_keys)

    # Remove the passed keys from the class keys.
    if keys[0] == "not":
        return {key for key in class_keys if key not in keys[1:]}
    # Check if all passed keys are valid
    if invalid_keys := set(keys) - class_keys:
        raise ValueError(f"Invalid keys: {invalid_keys}",)
    # Only return specified keys that are in class_keys
    return {key for key in keys if key in class_keys}


def generate_random_hex(length: int = 17) -> str:
    """Generate a random hex string
    Args:
        length (int, optional): Length of the hex string. Defaults to 17.
    Returns:
        str: Random hex string
    """
    return secrets.token_hex(length)


def random_int(min: int, max: int) -> int:
    """Generate a random integer
    Args:
        min (int): Minimum value
        max (int): Maximum value
    Returns:
        int: Random integer
    """
    return secrets.randbelow(max - min) + min


if __name__ == "__main__":
    logging.basicConfig(
        format=
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",)

log = logging.getLogger(__name__)


def logger(is_timed: bool):
    """Logger decorator
    Args:
        is_timed (bool): Whether to include function running time in exit log
    Returns:
        _type_: decorated function
    """

    def decorator(func):
        wraps(func)

        def wrapper(*args, **kwargs):
            log.debug(
                "Entering %s with args %s and kwargs %s",
                func.__name__,
                args,
                kwargs,
            )
            start = time.time()
            out = func(*args, **kwargs)
            end = time.time()
            if is_timed:
                log.debug(
                    "Exiting %s with return value %s. Took %s seconds.",
                    func.__name__,
                    out,
                    end - start,
                )
            else:
                log.debug("Exiting %s with return value %s", func.__name__, out)
            return out

        return wrapper

    return decorator
