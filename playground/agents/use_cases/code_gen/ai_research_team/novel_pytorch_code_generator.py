from swarms.models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field
from swarms import create_file_in_folder
from swarms.tools.prebuilt.code_executor import CodeExecutor
from swarms.utils.loguru_logger import logger
import threading


code_executor = CodeExecutor()


class ModelSpec(BaseModel):
    novel_algorithm_name: str = Field(
        ...,
        description="The name of the novel AI algorithm",
    )
    mathamatical_formulation: str = Field(
        ...,
        description="The mathamatical theortical formulation of the new model",
    )
    model_code: str = Field(
        ...,
        description="The code for the all-new model architecture in PyTorch, Add docs, and write clean code",
    )


# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're an expert model engineer like Lucidrains, you write world-class PHD level code for deep learning models. You're purpose is to create a novel deep learning model for a research paper. You need to provide the name of the model, the mathamatical formulation, and the code for the model architecture in PyTorch. Write clean and concise code that is easy to understand and implement. Write production-grade pytorch code, add types, and documentation. Make sure you track tensorshapes to not forget and write great pytorch code. Be creative and create models that have never been contemplated before",
    max_tokens=5000,
    temperature=0.6,
    base_model=ModelSpec,
    parallel_tool_calls=False,
)


def clean_model_code(model_code_str: str):
    # Remove extra escape characters and newlines
    cleaned_code = model_code_str.replace("\\n", "\n").replace("\\'", "'")

    # Remove unnecessary leading and trailing whitespaces
    cleaned_code = cleaned_code.strip()

    return cleaned_code


# for i in range(50):
#     # The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
#     out = model.run(
#         "Create an entirely new neural network operation aside from convolutions and the norm, write clean code and explain step by step"
#     )
#     name = out["novel_algorithm_name"]
#     logger.info(f"Generated code for novel model {i}:")

#     # Parse the 3 rows of the output || 0: novel_algorithm_name, 1: mathamatical_formulation, 2: model_code
#     out = out["model_code"]
#     out = clean_model_code(out)
#     logger.info(f"Cleansed code for novel model {i}:")

#     # Save the generated code to a file
#     create_file_in_folder("new_models", f"{name}.py", out)
#     logger.info(f"Saved code for novel model {i} to file:")

#     # # Execute the generated code
#     # logger.info(f"Executing code for novel model {i}:")
#     # test = code_executor.execute(out)
#     # logger.info(f"Executed code for novel model {i}: {test}")


# def execute_code_and_retry(code: str) -> str:
#     run = code_executor.execute(code)

#     if "error" in run:
#         logger.error(f"Error in code execution: {run}")


def generate_and_execute_model(i):
    # The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
    out = model.run(
        "Create an entirely new model architecture by blending backbones like attention, lstms, rnns, and ssms all into one novel architecture. Provide alternative model architectures to transformers, ssms, convnets, lstms, and more. Be creative and don't work on architectures that have been done before. The goal is to create new-ultra high performance nets"
    )
    name = out["novel_algorithm_name"]
    theory = out["mathamatical_formulation"]
    code = out["model_code"]
    logger.info(f"Generated code for novel model {name}:")

    # Parse the 3 rows of the output || 0: novel_algorithm_name, 1: mathamatical_formulation, 2: model_code
    code = clean_model_code(code)
    logger.info(f"Cleansed code for novel model {i}:")

    # Save the generated code to a file
    create_file_in_folder("new_models", f"{name}.py", code)
    logger.info(f"Saved code for novel model {i} to file:")

    # Execute the generated code
    test = code_executor.execute(code)

    if "error" in test:
        logger.error(f"Error in code execution: {test}")

        # Retry executing the code
        model.run(
            f"Recreate the code for the model: {name}, there was an error in the code you generated earlier execution: {code}. The theory was: {theory}"
        )

        name = out["novel_algorithm_name"]
        theory = out["mathamatical_formulation"]
        code = out["model_code"]

        # Clean the code
        code = clean_model_code(code)

        # Execute the code
        test = code_executor.execute(code)

        if "error" not in test:
            logger.info(
                f"Successfully executed code for novel model {name}"
            )
            create_file_in_folder("new_models", f"{name}.py", code)
        else:
            logger.error(f"Error in code execution: {test}")


# Create and start a new thread for each model
threads = []
for i in range(35):
    thread = threading.Thread(target=generate_and_execute_model, args=(i,))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()
