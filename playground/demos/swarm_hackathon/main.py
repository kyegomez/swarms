import concurrent
import csv
import os
from swarms import Gemini, Agent
from swarms.memory import ChromaDB
from dotenv import load_dotenv
from swarms.utils.parse_code import extract_code_from_markdown
from swarms.utils.file_processing import create_file
from swarms.utils.loguru_logger import logger

# Load ENV
load_dotenv()


gemini = Gemini(
    model_name="gemini-pro",
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
)

# memory
memory = ChromaDB(output_dir="swarm_hackathon")


def execute_concurrently(callable_functions: callable, max_workers=5):
    """
    Executes callable functions concurrently using multithreading.

    Parameters:
    - callable_functions: A list of tuples, each containing the callable function and its arguments.
      For example: [(function1, (arg1, arg2), {'kwarg1': val1}), (function2, (), {})]
    - max_workers: The maximum number of threads to use.

    Returns:
    - results: A list of results returned by the callable functions. If an error occurs in any function,
      the exception object will be placed at the corresponding index in the list.
    """
    results = [None] * len(callable_functions)

    def worker(fn, args, kwargs, index):
        try:
            result = fn(*args, **kwargs)
            results[index] = result
        except Exception as e:
            results[index] = e

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = []
        for i, (fn, args, kwargs) in enumerate(callable_functions):
            futures.append(executor.submit(worker, fn, args, kwargs, i))

        # Wait for all threads to complete
        concurrent.futures.wait(futures)

    return results


# Adjusting the function to extract specific column values
def extract_and_create_agents(csv_file_path: str, target_columns: list):
    """
    Reads a CSV file, extracts "Project Name" and "Lightning Proposal" for each row,
    creates an Agent for each, and adds it to the swarm network.

    Parameters:
    - csv_file_path: The path to the CSV file.
    - target_columns: A list of column names to extract values from.
    """
    agents = []
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            project_name = row[target_columns[0]]
            lightning_proposal = row[target_columns[1]]

            # Example of creating and adding an agent based on the project name and lightning proposal
            agent_name = f"{project_name} agent"
            print(agent_name)  # For demonstration

            # Create the agent
            logger.info("Creating agent...")
            agent = Agent(
                llm=gemini,
                max_loops=1,
                stopping_token="<DONE?>",
                sop=None,
                system_prompt=(
                    "Transform an app idea into a very simple python"
                    " app in markdown. Return all the python code in"
                    " a single markdown file."
                ),
                long_term_memory=memory,
            )

            # Log the agent
            logger.info(
                f"Agent created: {agent_name} with long term memory"
            )
            agents.append(agent)

            # Create the code for each project
            output = agent.run(
                (
                    f"Create the code for the {lightning_proposal} in"
                    " python and wrap it in markdown and return it"
                ),
                None,
            )
            print(output)
            # Parse the output
            output = extract_code_from_markdown(output)
            # Create the file
            output = create_file(output, f"{project_name}.py")

            # Log the project created
            logger.info(
                f"Project {project_name} created: {output} at file"
                f" path {project_name}.py"
            )
            print(output)

    return agents


# Specific columns to extract
target_columns = ["Project Name", "Lightning Proposal "]

# Use the adjusted function
specific_column_values = extract_and_create_agents(
    "text.csv", target_columns
)

# Display the extracted column values
print(specific_column_values)


# Concurrently execute the function
output = execute_concurrently(
    [
        (extract_and_create_agents, ("text.csv", target_columns), {}),
    ],
    max_workers=5,
)
print(output)
