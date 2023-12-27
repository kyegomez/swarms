###### VERISON2
import inspect
import os
import threading
from zeta import OpenAIChat
from scripts.auto_tests_docs.docs import DOCUMENTATION_WRITER_SOP
from zeta.nn.modules._activations import (
    AccurateGELUActivation,
    ClippedGELUActivation,
    FastGELUActivation,
    GELUActivation,
    LaplaceActivation,
    LinearActivation,
    MishActivation,
    NewGELUActivation,
    PytorchGELUTanh,
    QuickGELUActivation,
    ReLUSquaredActivation,
)
from zeta.nn.modules.dense_connect import DenseBlock
from zeta.nn.modules.dual_path_block import DualPathBlock
from zeta.nn.modules.feedback_block import FeedbackBlock
from zeta.nn.modules.highway_layer import HighwayLayer
from zeta.nn.modules.multi_scale_block import MultiScaleBlock
from zeta.nn.modules.recursive_block import RecursiveBlock
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIChat(
    model_name="gpt-4",
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
        f"Class Name: {cls.__name__}\n\nDocumentation:\n{doc}\n\nSource"
        f" Code:\n{source}"
    )
    print(input_content)

    # Process with OpenAI model (assuming the model's __call__ method takes this input and returns processed content)
    processed_content = model(DOCUMENTATION_WRITER_SOP(input_content, "zeta"))

    doc_content = f"# {cls.__name__}\n\n{processed_content}\n"

    # Create the directory if it doesn't exist
    dir_path = "docs/zeta/nn/modules"
    os.makedirs(dir_path, exist_ok=True)

    # Write the processed documentation to a Markdown file
    file_path = os.path.join(dir_path, f"{cls.__name__.lower()}.md")
    with open(file_path, "w") as file:
        file.write(doc_content)


def main():
    classes = [
        DenseBlock,
        HighwayLayer,
        MultiScaleBlock,
        FeedbackBlock,
        DualPathBlock,
        RecursiveBlock,
        PytorchGELUTanh,
        NewGELUActivation,
        GELUActivation,
        FastGELUActivation,
        QuickGELUActivation,
        ClippedGELUActivation,
        AccurateGELUActivation,
        MishActivation,
        LinearActivation,
        LaplaceActivation,
        ReLUSquaredActivation,
    ]

    threads = []
    for cls in classes:
        thread = threading.Thread(target=process_documentation, args=(cls,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Documentation generated in 'docs/zeta/nn/modules' directory.")


if __name__ == "__main__":
    main()
