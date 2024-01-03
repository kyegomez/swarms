###### VERISON2
import inspect
import os
import threading

from dotenv import load_dotenv

from scripts.auto_tests_docs.docs import DOCUMENTATION_WRITER_SOP
from swarms import OpenAIChat

##########
from swarms.structs.task import Task
from swarms.structs.swarm_net import SwarmNetwork
from swarms.structs.nonlinear_workflow import NonlinearWorkflow
from swarms.structs.recursive_workflow import RecursiveWorkflow
from swarms.structs.groupchat import GroupChat, GroupChatManager
from swarms.structs.base_workflow import BaseWorkflow
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.base import BaseStructure
from swarms.structs.schemas import (
    Artifact,
    ArtifactUpload,
    StepInput,
    TaskInput,
)

####################
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIChat(
    openai_api_key=api_key,
    max_tokens=4000,
)


def process_documentation(cls):
    """
    Process the documentation for a given class using OpenAI model and save it in a Markdown file.
    """
    doc = inspect.getdoc(cls)
    source = inspect.getsource(cls)
    input_content = (
        "Class Name:"
        f" {cls.__name__}\n\nDocumentation:\n{doc}\n\nSource"
        f" Code:\n{source}"
    )

    # Process with OpenAI model (assuming the model's __call__ method takes this input and returns processed content)
    processed_content = model(
        DOCUMENTATION_WRITER_SOP(input_content, "swarms.structs")
    )

    # doc_content = f"# {cls.__name__}\n\n{processed_content}\n"
    doc_content = f"{processed_content}\n"

    # Create the directory if it doesn't exist
    dir_path = "docs/swarms/structs"
    os.makedirs(dir_path, exist_ok=True)

    # Write the processed documentation to a Markdown file
    file_path = os.path.join(dir_path, f"{cls.__name__.lower()}.md")
    with open(file_path, "w") as file:
        file.write(doc_content)

    print(f"Documentation generated for {cls.__name__}.")


def main():
    classes = [
        Task,
        SwarmNetwork,
        NonlinearWorkflow,
        RecursiveWorkflow,
        GroupChat,
        GroupChatManager,
        BaseWorkflow,
        ConcurrentWorkflow,
        BaseStructure,
        Artifact,
        ArtifactUpload,
        StepInput,
        TaskInput,
    ]
    threads = []
    for cls in classes:
        thread = threading.Thread(
            target=process_documentation, args=(cls,)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Documentation generated in 'swarms.structs' directory.")


if __name__ == "__main__":
    main()
