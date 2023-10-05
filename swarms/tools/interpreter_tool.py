import os
import interpreter


def compile(task: str):
    """
    Open Interpreter lets LLMs run code (Python, Javascript, Shell, and more) locally. You can chat with Open Interpreter through a ChatGPT-like interface in your terminal by running $ interpreter after installing.

    This provides a natural-language interface to your computer's general-purpose capabilities:

    Create and edit photos, videos, PDFs, etc.
    Control a Chrome browser to perform research
    Plot, clean, and analyze large datasets
    ...etc.
    ⚠️ Note: You'll be asked to approve code before it's run.
    """

    task = interpreter.chat(task, return_messages=True)
    interpreter.chat()
    interpreter.reset(task)

    os.environ["INTERPRETER_CLI_AUTO_RUN"] = True
    os.environ["INTERPRETER_CLI_FAST_MODE"] = True
    os.environ["INTERPRETER_CLI_DEBUG"] = True
