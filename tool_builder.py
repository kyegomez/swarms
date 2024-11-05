import os
from pydantic import BaseModel, Field
from swarm_models import OpenAIFunctionCaller
from dotenv import load_dotenv
from typing import Any
from swarms.utils.loguru_logger import logger
from swarms.tools.prebuilt.code_executor import CodeExecutor

load_dotenv()


class Tool(BaseModel):
    id: str = Field(
        description="A unique identifier for the task. This should be a short, descriptive name that captures the main purpose of the task. Use - to separate words and make it lowercase."
    )
    plan: str = Field(
        description="The comprehensive plan detailing how the task will accomplish the given task. This should include the high-level strategy, key milestones, and expected outcomes. The plan should clearly articulate what the overall goal is, what success looks like, and how progress will be measured throughout execution."
    )
    failures_prediction: str = Field(
        description="A thorough analysis of potential failure modes and mitigation strategies. This should identify technical risks, edge cases, error conditions, and possible points of failure in the task. For each identified risk, include specific preventive measures, fallback approaches, and recovery procedures to ensure robustness and reliability."
    )
    rationale: str = Field(
        description="The detailed reasoning and justification for why this specific task design is optimal for the given task. This should explain the key architectural decisions, tradeoffs considered, alternatives evaluated, and why this approach best satisfies the requirements. Include both technical and business factors that influenced the design."
    )
    code: str = Field(
        description="Generate the code for the task. This should be a python function that takes in a task and returns a result. The code should be a complete and working implementation of the task. Include all necessary imports and dependencies and add types, docstrings, and comments to the code. Make sure the main code executes successfully. No placeholders or comments. Make sure the main function executes successfully."
    )


def setup_model(base_model: BaseModel = Tool):
    model = OpenAIFunctionCaller(
        system_prompt="""You are an expert Python developer specializing in building reliable API integrations and developer tools. Your role is to generate production-ready code that follows best practices for API interactions and tool development.

        When given a task, you will:
        1. Design robust error handling and retry mechanisms for API calls
        2. Implement proper authentication and security measures
        3. Structure code for maintainability and reusability
        4. Add comprehensive logging and monitoring
        5. Include detailed type hints and documentation
        6. Write unit tests to verify functionality

        Your code should follow these principles:
        - Use modern Python features and idioms
        - Handle rate limits and API quotas gracefully
        - Validate inputs and outputs thoroughly
        - Follow security best practices for API keys and secrets
        - Include clear error messages and debugging info
        - Be well-documented with docstrings and comments
        - Use appropriate design patterns
        - Follow PEP 8 style guidelines

        The generated code should be complete, tested, and ready for production use. Include all necessary imports, error handling, and helper functions.
        """,
        base_model=base_model,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5,
    )
    return model


def generate_tool(task: str) -> Any:
    model = setup_model()
    response = model.run(task)
    logger.info(f"Response: {response}")

    # If response is a dict, get code directly
    if isinstance(response, dict):
        # return response.get("code", "")
        code = response.get("code", "")
        logger.info(f"Code: {code}")
        return code
    # If response is a Tool object, access code attribute
    elif isinstance(response, Tool):
        code = response.code
        logger.info(f"Code: {code}")
        return code
    # If response is a string (raw code)
    elif isinstance(response, str):
        code = response
        logger.info(f"Code: {code}")
        return code
    logger.error(f"Unexpected response type: {type(response)}")
    return ""


def execute_generated_code(code: str) -> Any:
    """
    Attempts to execute the generated Python code, handling errors and retrying if necessary.

    Args:
        code (str): The Python code to be executed.

    Returns:
        Any: Output of the code execution, or error details if execution fails.
    """
    logger.info("Starting code execution")
    try:
        exec_namespace = {}
        exec(code, exec_namespace)

        # Check for any callable functions in the namespace
        main_function = None
        for item in exec_namespace.values():
            if callable(item) and not item.__name__.startswith("__"):
                main_function = item
                break

        if main_function:
            result = main_function()
            logger.info(
                f"Code execution successful. Function result: {result}"
            )
            return result
        elif "result" in exec_namespace:
            logger.info(
                f"Code execution successful. Result variable: {exec_namespace['result']}"
            )
            return exec_namespace["result"]
        else:
            logger.warning(
                "Code execution completed but no result found"
            )
            return "No result or function found in executed code."
    except Exception as e:
        logger.error(
            f"Code execution failed with error: {str(e)}",
            exc_info=True,
        )
        return e


def retry_until_success(task: str, max_retries: int = 5):
    """
    Generates and executes code until the execution is successful.

    Args:
        task (str): Task description to generate the required code.
    """
    attempts = 0

    while attempts < max_retries:
        logger.info(f"Attempt {attempts + 1} of {max_retries}")
        tool = generate_tool(task)
        logger.debug(f"Generated code:\n{tool}")

        # result = execute_generated_code(tool)
        result = CodeExecutor().execute(code=tool)
        logger.info(f"Result: {result}")

        if isinstance(result, Exception):
            logger.error(
                f"Attempt {attempts + 1} failed: {str(result)}"
            )
            print("Retrying with updated code...")
            attempts += 1
        else:
            logger.info(
                f"Success on attempt {attempts + 1}. Result: {result}"
            )
            print(f"Code executed successfully: {result}")
            break
    else:
        logger.error("Max retries reached. Execution failed.")
        print("Max retries reached. Execution failed.")


# Usage
retry_until_success(
    "Write a function to fetch and display weather information from a given API."
)
