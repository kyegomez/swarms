import inspect
import os
import threading

from dotenv import load_dotenv
from scripts.auto_tests_docs.docs import DOCUMENTATION_WRITER_SOP
from swarms import OpenAIChat
from swarms.structs.agent import Agent
from swarms.structs.autoscaler import AutoScaler
from swarms.structs.base import BaseStructure
from swarms.structs.base_swarm import AbstractSwarm
from swarms.structs.base_workflow import BaseWorkflow
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.conversation import Conversation
from swarms.structs.groupchat import GroupChat, GroupChatManager
from swarms.structs.model_parallizer import ModelParallelizer
from swarms.structs.multi_agent_collab import MultiAgentCollaboration
from swarms.structs.nonlinear_workflow import NonlinearWorkflow
from swarms.structs.recursive_workflow import RecursiveWorkflow
from swarms.structs.schemas import (
    Artifact,
    ArtifactUpload,
    StepInput,
    TaskInput,
)
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.swarm_net import SwarmNetwork
from swarms.structs.utils import (
    distribute_tasks,
    extract_key_from_json,
    extract_tokens_from_text,
    find_agent_by_id,
    find_token_in_text,
    parse_tasks,
)


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIChat(
    model_name="gpt-4-1106-preview",
    openai_api_key=api_key,
    max_tokens=4000,
)


def process_documentation(
    item,
    module: str = "swarms.structs",
    docs_folder_path: str = "docs/swarms/structs",
):
    """
    Process the documentation for a given class or function using OpenAI model and save it in a Python file.
    """
    doc = inspect.getdoc(item)
    source = inspect.getsource(item)
    is_class = inspect.isclass(item)
    item_type = "Class Name" if is_class else "Name"
    input_content = (
        f"{item_type}:"
        f" {item.__name__}\n\nDocumentation:\n{doc}\n\nSource"
        f" Code:\n{source}"
    )

    # Process with OpenAI model
    processed_content = model(
        DOCUMENTATION_WRITER_SOP(input_content, module)
    )

    doc_content = f"# {item.__name__}\n\n{processed_content}\n"

    # Create the directory if it doesn't exist
    dir_path = docs_folder_path
    os.makedirs(dir_path, exist_ok=True)

    # Write the processed documentation to a Python file
    file_path = os.path.join(dir_path, f"{item.__name__.lower()}.md")
    with open(file_path, "w") as file:
        file.write(doc_content)

    print(
        f"Processed documentation for {item.__name__}. at {file_path}"
    )


def main(module: str = "docs/swarms/structs"):
    items = [
        Agent,
        SequentialWorkflow,
        AutoScaler,
        Conversation,
        TaskInput,
        Artifact,
        ArtifactUpload,
        StepInput,
        SwarmNetwork,
        ModelParallelizer,
        MultiAgentCollaboration,
        AbstractSwarm,
        GroupChat,
        GroupChatManager,
        parse_tasks,
        find_agent_by_id,
        distribute_tasks,
        find_token_in_text,
        extract_key_from_json,
        extract_tokens_from_text,
        ConcurrentWorkflow,
        RecursiveWorkflow,
        NonlinearWorkflow,
        BaseWorkflow,
        BaseStructure,
    ]

    threads = []
    for item in items:
        thread = threading.Thread(
            target=process_documentation, args=(item,)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print(f"Documentation generated in {module} directory.")


if __name__ == "__main__":
    main()
