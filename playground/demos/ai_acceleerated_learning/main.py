import concurrent
import csv
from swarms import Agent, OpenAIChat
from swarms.memory import ChromaDB
from dotenv import load_dotenv
from swarms.utils.parse_code import extract_code_from_markdown
from swarms.utils.file_processing import create_file
from swarms.utils.loguru_logger import logger


# Load ENV
load_dotenv()

# Gemini
gemini = OpenAIChat()

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
    try:
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

                # Design agent
                logger.info("Creating design agent...")
                design_agent = Agent(
                    llm=gemini,
                    agent_name="Design Agent",
                    max_loops=1,
                    stopping_token="<DONE?>",
                    sop=None,
                    system_prompt=(
                        "Transform an app idea into step by step very"
                        " simple algorithmic psuedocode so it can be"
                        " implemented simply."
                    ),
                    long_term_memory=memory,
                )

                # Log the agent
                logger.info(
                    f"Code Agent created: {agent_name} with long term"
                    " memory"
                )
                agent = Agent(
                    llm=gemini,
                    agent_name=agent_name,
                    max_loops=1,
                    code_interpreter=True,
                    stopping_token="<DONE?>",
                    sop=None,
                    system_prompt=(
                        "Transform an app idea into a very simple"
                        " python app in markdown. Return all the"
                        " python code in a single markdown file."
                        " Return only code and nothing else."
                    ),
                    long_term_memory=memory,
                )

                # Testing agent
                logger.info(f"Testing_agent agent: {agent_name}")
                agent = Agent(
                    llm=gemini,
                    agent_name=agent_name + " testing",
                    max_loops=1,
                    stopping_token="<DONE?>",
                    sop=None,
                    system_prompt=(
                        "Create unit tests using pytest based on the"
                        " code you see, only return unit test code in"
                        " python using markdown, only return the code"
                        " and nothing else."
                    ),
                    long_term_memory=memory,
                )

                # Log the agent
                logger.info(
                    f"Agent created: {agent_name} with long term" " memory"
                )
                agents.append(agent)

                # Design agent
                design_agent_output = design_agent.run(
                    (
                        "Create the algorithmic psuedocode for the"
                        f" {lightning_proposal} in markdown and"
                        " return it"
                    ),
                    None,
                )

                logger.info(
                    "Algorithmic psuedocode created:"
                    f" {design_agent_output}"
                )

                # Create the code for each project
                output = agent.run(
                    (
                        "Create the code for the"
                        f" {lightning_proposal} in python using the"
                        " algorithmic psuedocode"
                        f" {design_agent_output} and wrap it in"
                        " markdown and return it"
                    ),
                    None,
                )
                print(output)
                # Parse the output
                output = extract_code_from_markdown(output)
                # Create the file
                output = create_file(output, f"{project_name}.py")

                # Testing agent
                testing_agent_output = agent.run(
                    (
                        "Create the unit tests for the"
                        f" {lightning_proposal} in python using the"
                        f" code {output} and wrap it in markdown and"
                        " return it"
                    ),
                    None,
                )
                print(testing_agent_output)

                # Parse the output
                testing_agent_output = extract_code_from_markdown(
                    testing_agent_output
                )
                # Create the file
                testing_agent_output = create_file(
                    testing_agent_output, f"test_{project_name}.py"
                )

                # Log the project created
                logger.info(
                    f"Project {project_name} created: {output} at"
                    f" file path {project_name}.py"
                )
                print(output)

                # Log the unit tests created
                logger.info(
                    f"Unit tests for {project_name} created:"
                    f" {testing_agent_output} at file path"
                    f" test_{project_name}.py"
                )

                print(
                    f"Agent {agent_name} created and added to the"
                    " swarm network"
                )

        return agents

    except Exception as e:
        logger.error(
            "An error occurred while extracting and creating"
            f" agents: {e}"
        )
        return None


# CSV
csv_file = "presentation.csv"

# Specific columns to extract
target_columns = ["Project Name", "Project Description"]

# Use the adjusted function
specific_column_values = extract_and_create_agents(
    csv_file, target_columns
)

# Display the extracted column values
print(specific_column_values)


# Concurrently execute the function
logger.info(
    "Concurrently executing the swarm for each hackathon project..."
)
output = execute_concurrently(
    [
        (extract_and_create_agents, (csv_file, target_columns), {}),
    ],
    max_workers=5,
)
print(output)
