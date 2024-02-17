import os


from swarms import (
    OpenAIChat,
    Conversation,
    detect_markdown,
    extract_code_from_markdown,
)

from swarms.tools.code_executor import CodeExecutor

from swarms.utils import load_environment

conv = Conversation(
    autosave=False,
    time_enabled=True,
)

# Load the environment variables
api_key = load_environment()

# Initialize the language model
llm = OpenAIChat(openai_api_key=api_key)


# Run the language model in a loop
def interactive_conversation(llm, iters: int = 10):
    conv = Conversation()
    for i in range(iters):
        user_input = input("User: ")
        conv.add("user", user_input)

        if user_input.lower() == "quit":
            break

        task = (
            conv.return_history_as_string()
        )  # Get the conversation history

        # Run the language model
        out = llm(task)
        conv.add("assistant", out)
        print(
            f"Assistant: {out}",
        )

        # Code Interpreter
        if detect_markdown(out):
            code = extract_code_from_markdown(out)
            if code:
                print(f"Code: {code}")
                executor = CodeExecutor()
                out = executor.run(code)
                conv.add("assistant", out)
                # print(f"Assistant: {out}")

        conv.display_conversation()
        # conv.export_conversation("conversation.txt")


# Replace with your LLM instance
interactive_conversation(llm)
