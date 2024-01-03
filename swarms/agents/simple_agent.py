from swarms.structs.conversation import Conversation
from swarms.models.base_llm import AbstractLLM


# Run the language model in a loop for n iterations
def SimpleAgent(
    llm: AbstractLLM = None, iters: int = 10, *args, **kwargs
):
    """Simple agent conversation

    Args:
        llm (_type_): _description_
        iters (int, optional): _description_. Defaults to 10.

    Example:
        >>> from swarms.models import GPT2LM
        >>> from swarms.agents import SimpleAgent
        >>> llm = GPT2LM()
        >>> SimpleAgent(llm, iters=10)
    """
    try:
        conv = Conversation(*args, **kwargs)
        for i in range(iters):
            user_input = input("User: ")
            conv.add("user", user_input)
            if user_input.lower() == "quit":
                break
            task = (
                conv.return_history_as_string()
            )  # Get the conversation history
            out = llm(task)
            conv.add("assistant", out)
            print(
                f"Assistant: {out}",
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
