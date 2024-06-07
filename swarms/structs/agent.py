import asyncio
import concurrent.futures
import json
import logging
import os
import random
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from loguru import logger
from pydantic import BaseModel
from termcolor import colored

from swarms.memory.base_vectordb import BaseVectorDatabase
from swarms.prompts.agent_system_prompts import AGENT_SYSTEM_PROMPT_3
from swarms.prompts.aot_prompt import algorithm_of_thoughts_sop
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)
from swarms.structs.conversation import Conversation
from swarms.structs.yaml_model import YamlModel
from swarms.telemetry.user_utils import get_user_device_data
from swarms.tools.prebuilt.code_interpreter import (
    SubprocessCodeInterpreter,
)
from swarms.tools.pydantic_to_json import (
    multi_base_model_to_openai_function,
)
from swarms.utils.data_to_text import data_to_text
from swarms.utils.parse_code import extract_code_from_markdown
from swarms.utils.pdf_to_text import pdf_to_text
from swarms.tools.py_func_to_openai_func_str import (
    get_openai_function_schema_from_func,
)
from swarms.structs.base_structure import BaseStructure
from swarms.prompts.tools import tool_sop_prompt
from swarms.tools.func_calling_utils import (
    pydantic_model_to_json_str,
    prepare_output_for_output_model,
)
from swarms.tools.tool_parse_exec import parse_and_execute_json


# Utils
# Custom stopping condition
def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "stop" in response.lower()


# Parse done token
def parse_done_token(response: str) -> bool:
    """Parse the response to see if the done token is present"""
    return "<DONE>" in response


# Agent ID generator
def agent_id():
    """Generate an agent id"""
    return str(uuid.uuid4())


# Task ID generator
def task_id():
    """
    Generate a unique task ID.

    Returns:
        str: A string representation of a UUID.
    """
    return str(uuid.uuid4())


def exists(val):
    return val is not None


# Step ID generator
def step_id():
    return str(uuid.uuid1())


# Agent output types
agent_output_type = Union[BaseModel, dict, str]
ToolUsageType = Union[BaseModel, Dict[str, Any]]


def retrieve_tokens(text, num_tokens):
    """
    Retrieve a specified number of tokens from a given text.

    Parameters:
    text (str): The input text string.
    num_tokens (int): The number of tokens to retrieve.

    Returns:
    str: A string containing the specified number of tokens from the input text.
    """
    # Initialize an empty list to store tokens
    tokens = []
    token_count = 0

    # Split the text into words while counting tokens
    for word in text.split():
        tokens.append(word)
        token_count += 1
        if token_count == num_tokens:
            break

    # Join the selected tokens back into a string
    result = " ".join(tokens)

    return result


# [FEAT][AGENT]
class Agent(BaseStructure):
    """
    Agent is the backbone to connect LLMs with tools and long term memory. Agent also provides the ability to
    ingest any type of docs like PDFs, Txts, Markdown, Json, and etc for the agent. Here is a list of features.

    Args:
        llm (Any): The language model to use
        template (str): The template to use
        max_loops (int): The maximum number of loops to run
        stopping_condition (Callable): The stopping condition to use
        loop_interval (int): The loop interval
        retry_attempts (int): The number of retry attempts
        retry_interval (int): The retry interval
        return_history (bool): Return the history
        stopping_token (str): The stopping token
        dynamic_loops (bool): Enable dynamic loops
        interactive (bool): Enable interactive mode
        dashboard (bool): Enable dashboard
        agent_name (str): The name of the agent
        agent_description (str): The description of the agent
        system_prompt (str): The system prompt
        tools (List[BaseTool]): The tools to use
        dynamic_temperature_enabled (bool): Enable dynamic temperature
        sop (str): The standard operating procedure
        sop_list (List[str]): The standard operating procedure list
        saved_state_path (str): The path to the saved state
        autosave (bool): Autosave the state
        context_length (int): The context length
        user_name (str): The user name
        self_healing_enabled (bool): Enable self healing
        code_interpreter (bool): Enable code interpreter
        multi_modal (bool): Enable multimodal
        pdf_path (str): The path to the pdf
        list_of_pdf (str): The list of pdf
        tokenizer (Any): The tokenizer
        memory (BaseVectorDatabase): The memory
        preset_stopping_token (bool): Enable preset stopping token
        traceback (Any): The traceback
        traceback_handlers (Any): The traceback handlers
        streaming_on (bool): Enable streaming

    Methods:
        run: Run the agent
        run_concurrent: Run the agent concurrently
        bulk_run: Run the agent in bulk
        save: Save the agent
        load: Load the agent
        validate_response: Validate the response
        print_history_and_memory: Print the history and memory
        step: Step through the agent
        graceful_shutdown: Gracefully shutdown the agent
        run_with_timeout: Run the agent with a timeout
        analyze_feedback: Analyze the feedback
        undo_last: Undo the last response
        add_response_filter: Add a response filter
        apply_response_filters: Apply the response filters
        filtered_run: Run the agent with filtered responses
        interactive_run: Run the agent in interactive mode
        streamed_generation: Stream the generation of the response
        save_state: Save the state
        load_state: Load the state
        truncate_history: Truncate the history
        add_task_to_memory: Add the task to the memory
        add_message_to_memory: Add the message to the memory
        add_message_to_memory_and_truncate: Add the message to the memory and truncate
        print_dashboard: Print the dashboard
        loop_count_print: Print the loop count
        streaming: Stream the content
        _history: Generate the history
        _dynamic_prompt_setup: Setup the dynamic prompt
        run_async: Run the agent asynchronously
        run_async_concurrent: Run the agent asynchronously and concurrently
        run_async_concurrent: Run the agent asynchronously and concurrently
        construct_dynamic_prompt: Construct the dynamic prompt
        construct_dynamic_prompt: Construct the dynamic prompt


    Examples:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import Agent
    >>> llm = OpenAIChat()
    >>> agent = Agent(llm=llm, max_loops=1)
    >>> response = agent.run("Generate a report on the financials.")
    >>> print(response)
    >>> # Generate a report on the financials.

    """

    def __init__(
        self,
        id: Optional[str] = agent_id,
        llm: Optional[Any] = None,
        template: Optional[str] = None,
        max_loops: Optional[int] = 1,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: Optional[int] = 0,
        retry_attempts: Optional[int] = 3,
        retry_interval: Optional[int] = 1,
        return_history: Optional[bool] = False,
        stopping_token: Optional[str] = None,
        dynamic_loops: Optional[bool] = False,
        interactive: Optional[bool] = False,
        dashboard: Optional[bool] = False,
        agent_name: Optional[str] = "swarm-worker-01",
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = AGENT_SYSTEM_PROMPT_3,
        # TODO: Change to callable, then parse the callable to a string
        tools: List[Callable] = None,
        dynamic_temperature_enabled: Optional[bool] = False,
        sop: Optional[str] = None,
        sop_list: Optional[List[str]] = None,
        saved_state_path: Optional[str] = None,
        autosave: Optional[bool] = False,
        context_length: Optional[int] = 8192,
        user_name: Optional[str] = "Human:",
        self_healing_enabled: Optional[bool] = False,
        code_interpreter: Optional[bool] = False,
        multi_modal: Optional[bool] = None,
        pdf_path: Optional[str] = None,
        list_of_pdf: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        long_term_memory: Optional[BaseVectorDatabase] = None,
        preset_stopping_token: Optional[bool] = False,
        traceback: Optional[Any] = None,
        traceback_handlers: Optional[Any] = None,
        streaming_on: Optional[bool] = False,
        docs: List[str] = None,
        docs_folder: Optional[str] = None,
        verbose: Optional[bool] = False,
        parser: Optional[Callable] = None,
        best_of_n: Optional[int] = None,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callable]] = None,
        logger_handler: Optional[Any] = sys.stderr,
        search_algorithm: Optional[Callable] = None,
        logs_to_filename: Optional[str] = None,
        evaluator: Optional[Callable] = None,  # Custom LLM or agent
        output_json: Optional[bool] = False,
        stopping_func: Optional[Callable] = None,
        custom_loop_condition: Optional[Callable] = None,
        sentiment_threshold: Optional[
            float
        ] = None,  # Evaluate on output using an external model
        custom_exit_command: Optional[str] = "exit",
        sentiment_analyzer: Optional[Callable] = None,
        limit_tokens_from_string: Optional[Callable] = None,
        # [Tools]
        custom_tools_prompt: Optional[Callable] = None,
        tool_schema: ToolUsageType = None,
        output_type: agent_output_type = None,
        function_calling_type: str = "json",
        output_cleaner: Optional[Callable] = None,
        function_calling_format_type: Optional[str] = "OpenAI",
        list_base_models: Optional[List[BaseModel]] = None,
        metadata_output_type: str = "json",
        state_save_file_type: str = "json",
        chain_of_thoughts: bool = False,
        algorithm_of_thoughts: bool = False,
        tree_of_thoughts: bool = False,
        tool_choice: str = "auto",
        execute_tool: bool = False,
        rules: str = None,
        planning: Optional[str] = False,
        planning_prompt: Optional[str] = None,
        device: str = None,
        custom_planning_prompt: str = None,
        memory_chunk_size: int = 2000,
        agent_ops_on: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.id = id
        self.llm = llm
        self.template = template
        self.max_loops = max_loops
        self.stopping_condition = stopping_condition
        self.loop_interval = loop_interval
        self.retry_attempts = retry_attempts
        self.retry_interval = retry_interval
        self.task = None
        self.stopping_token = stopping_token
        self.interactive = interactive
        self.dashboard = dashboard
        self.return_history = return_history
        self.dynamic_temperature_enabled = dynamic_temperature_enabled
        self.dynamic_loops = dynamic_loops
        self.user_name = user_name
        self.context_length = context_length
        self.sop = sop
        self.sop_list = sop_list
        self.tools = tools
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.saved_state_path = saved_state_path
        self.saved_state_path = f"{self.agent_name}_state.json"
        self.autosave = autosave
        self.response_filters = []
        self.self_healing_enabled = self_healing_enabled
        self.code_interpreter = code_interpreter
        self.multi_modal = multi_modal
        self.pdf_path = pdf_path
        self.list_of_pdf = list_of_pdf
        self.tokenizer = tokenizer
        self.long_term_memory = long_term_memory
        self.preset_stopping_token = preset_stopping_token
        self.traceback = traceback
        self.traceback_handlers = traceback_handlers
        self.streaming_on = streaming_on
        self.docs = docs
        self.docs_folder = docs_folder
        self.verbose = verbose
        self.parser = parser
        self.best_of_n = best_of_n
        self.callback = callback
        self.metadata = metadata
        self.callbacks = callbacks
        self.logger_handler = logger_handler
        self.search_algorithm = search_algorithm
        self.logs_to_filename = logs_to_filename
        self.evaluator = evaluator
        self.output_json = output_json
        self.stopping_func = stopping_func
        self.custom_loop_condition = custom_loop_condition
        self.sentiment_threshold = sentiment_threshold
        self.custom_exit_command = custom_exit_command
        self.sentiment_analyzer = sentiment_analyzer
        self.limit_tokens_from_string = limit_tokens_from_string
        self.tool_schema = tool_schema
        self.output_type = output_type
        self.function_calling_type = function_calling_type
        self.output_cleaner = output_cleaner
        self.function_calling_format_type = function_calling_format_type
        self.list_base_models = list_base_models
        self.metadata_output_type = metadata_output_type
        self.state_save_file_type = state_save_file_type
        self.chain_of_thoughts = chain_of_thoughts
        self.algorithm_of_thoughts = algorithm_of_thoughts
        self.tree_of_thoughts = tree_of_thoughts
        self.tool_choice = tool_choice
        self.execute_tool = execute_tool
        self.planning = planning
        self.planning_prompt = planning_prompt
        self.device = device
        self.custom_planning_prompt = custom_planning_prompt
        self.rules = rules
        self.custom_tools_prompt = custom_tools_prompt
        self.memory_chunk_size = memory_chunk_size
        self.agent_ops_on = agent_ops_on

        # Name
        self.name = agent_name
        self.description = agent_description
        # Agentic stuff
        self.reply = ""
        self.question = None
        self.answer = ""

        # The max_loops will be set dynamically if the dynamic_loop
        if self.dynamic_loops is True:
            logger.info("Dynamic loops enabled")
            self.max_loops = "auto"

        # If multimodal = yes then set the sop to the multimodal sop
        if self.multi_modal is True:
            self.sop = MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1

        # Memory
        self.feedback = []

        # If the preset stopping token is enabled then set the stopping token to the preset stopping token
        if preset_stopping_token is not None:
            self.stopping_token = "<DONE>"

        # If the system prompt is provided then set the system prompt
        # Initialize the short term memory
        self.short_memory = Conversation(
            system_prompt=system_prompt,
            time_enabled=True,
            user=user_name,
            rules=rules,
            *args,
            **kwargs,
        )

        # If the docs exist then ingest the docs
        if exists(self.docs):
            self.ingest_docs(self.docs)

        # If docs folder exists then get the docs from docs folder
        if exists(self.docs_folder):
            self.get_docs_from_doc_folders()

        if tools is not None:
            logger.info(
                "Tools provided make sure the functions have documentation ++ type hints, otherwise tool execution won't be reliable."
            )
            # Add the tool prompt to the memory
            self.short_memory.add(role="System", content=tool_sop_prompt())

            # Print number of tools
            logger.info("Tools granted, initializing tool protocol.")
            logger.info(f"Number of tools: {len(tools)}")

            # Transform the tools into an openai schema
            self.convert_tool_into_openai_schema()

            # Now create a function calling map for every tools
            self.function_map = {tool.__name__: tool for tool in tools}

        # Set the logger handler
        if exists(logger_handler):
            logger.add(
                f"{self.agent_name}.log",
                level="INFO",
                colorize=True,
                format=("<green>{time}</green> <level>{message}</level>"),
                backtrace=True,
                diagnose=True,
            )

        # If the tool types are provided
        if self.tool_schema is not None:
            # Log the tool schema
            logger.info(
                "Tool schema provided, Automatically converting to OpenAI function"
            )
            tool_schema_str = pydantic_model_to_json_str(
                self.tool_schema, indent=4
            )
            logger.info(f"Tool Schema: {tool_schema_str}")
            # Add the tool schema to the short memory
            self.short_memory.add(
                role=self.user_name, content=tool_schema_str
            )

        # If a list of tool schemas is provided
        if exists(self.list_base_models):
            logger.info(
                "List of tool schemas provided, Automatically converting to OpenAI function"
            )
            tool_schemas = multi_base_model_to_openai_function(
                self.list_base_models
            )

            # Convert the tool schemas to a string
            tool_schemas = json.dumps(tool_schemas, indent=4)

            # Add the tool schema to the short memory
            logger.info("Adding tool schema to short memory")
            self.short_memory.add(
                role=self.user_name, content=tool_schemas
            )

        # If the algorithm of thoughts is enabled then set the sop to the algorithm of thoughts
        if self.algorithm_of_thoughts is not False:
            self.short_memory.add(
                role=self.agent_name,
                content=algorithm_of_thoughts_sop(objective=self.task),
            )

        # Return the history
        if return_history is True:
            logger.info(f"Beginning of Agent {self.agent_name} History")
            logger.info(self.short_memory.return_history_as_string())
            logger.info(f"End of Agent {self.agent_name} History")

        # If the user inputs a list of strings for the sop then join them and set the sop
        if exists(self.sop_list):
            self.sop = "\n".join(self.sop_list)
            self.short_memory.add(role=self.user_name, content=self.sop)

        if exists(self.sop):
            self.short_memory.add(role=self.user_name, content=self.sop)

        # If the device is not provided then get the device data

        # if agent ops is enabled then import agent ops
        if agent_ops_on is True:
            try:
                from swarms.utils.agent_ops_check import (
                    try_import_agentops,
                )

                # Try importing agent ops
                logger.info(
                    "Agent Ops Initializing, ensure that you have the agentops API key and the pip package installed."
                )
                try_import_agentops()
            except ImportError:
                logger.error(
                    "Could not import agentops, try installing agentops: $ pip3 install agentops"
                )

    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt"""
        self.system_prompt = system_prompt

    def provide_feedback(self, feedback: str) -> None:
        """Allow users to provide feedback on the responses."""
        self.feedback.append(feedback)
        logging.info(f"Feedback received: {feedback}")

    def _check_stopping_condition(self, response: str) -> bool:
        """Check if the stopping condition is met."""
        try:
            if self.stopping_condition:
                return self.stopping_condition(response)
            return False
        except Exception as error:
            print(
                colored(
                    f"Error checking stopping condition: {error}",
                    "red",
                )
            )

    def dynamic_temperature(self):
        """
        1. Check the self.llm object for the temperature
        2. If the temperature is not present, then use the default temperature
        3. If the temperature is present, then dynamically change the temperature
        4. for every loop you can randomly change the temperature on a scale from 0.0 to 1.0
        """
        try:
            if hasattr(self.llm, "temperature"):
                # Randomly change the temperature attribute of self.llm object
                self.llm.temperature = random.uniform(0.0, 1.0)
                logger.info(f"Temperature: {self.llm.temperature}")
            else:
                # Use a default temperature
                self.llm.temperature = 0.7
        except Exception as error:
            print(
                colored(f"Error dynamically changing temperature: {error}")
            )

    def format_prompt(self, template, **kwargs: Any) -> str:
        """Format the template with the provided kwargs using f-string interpolation."""
        return template.format(**kwargs)

    def add_task_to_memory(self, task: str):
        """Add the task to the memory"""
        try:
            logger.info(f"Adding task to memory: {task}")
            self.short_memory.add(f"{self.user_name}: {task}")
        except Exception as error:
            print(colored(f"Error adding task to memory: {error}", "red"))

    # ############## TOKENIZER FUNCTIONS ##############
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return self.tokenizer.len(text)

    def tokens_per_second(self, text: str) -> float:
        """
        Calculates the number of tokens processed per second.

        Args:
            text (str): The input text to count tokens from.

        Returns:
            float: The number of tokens processed per second.
        """
        import time

        start_time = time.time()
        tokens = self.count_tokens(text)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return tokens / elapsed_time

    def time_to_generate(self, text: str) -> float:
        """
        Calculates the time taken to generate the output.

        Args:
            text (str): The input text to generate output from.

        Returns:
            float: The time taken to generate the output.
        """
        import time

        start_time = time.time()
        self.llm(text)
        end_time = time.time()
        return end_time - start_time

    # ############## TOKENIZER FUNCTIONS ##############

    def add_message_to_memory(self, message: str, *args, **kwargs):
        """Add the message to the memory"""
        try:
            logger.info(f"Adding message to memory: {message}")
            self.short_memory.add(
                role=self.agent_name, content=message, *args, **kwargs
            )
        except Exception as error:
            print(
                colored(f"Error adding message to memory: {error}", "red")
            )

    def add_message_to_memory_and_truncate(self, message: str):
        """Add the message to the memory and truncate"""
        self.short_memory[-1].append(message)
        self.truncate_history()

    def print_dashboard(self, task: str):
        """Print dashboard"""
        print(colored("Initializing Agent Dashboard...", "yellow"))

        print(
            colored(
                f"""
                Agent Dashboard
                --------------------------------------------

                Agent loop is initializing for {self.max_loops} with the following configuration:
                ----------------------------------------

                Agent Configuration:
                    Agent ID: {self.id}
                    Name: {self.agent_name}
                    Description: {self.agent_description}
                    Standard Operating Procedure: {self.sop}
                    System Prompt: {self.system_prompt} 
                    Task: {task}
                    Max Loops: {self.max_loops}
                    Stopping Condition: {self.stopping_condition}
                    Loop Interval: {self.loop_interval}
                    Retry Attempts: {self.retry_attempts}
                    Retry Interval: {self.retry_interval}
                    Interactive: {self.interactive}
                    Dashboard: {self.dashboard}
                    Dynamic Temperature: {self.dynamic_temperature_enabled}
                    Autosave: {self.autosave}
                    Saved State: {self.saved_state_path}

                ----------------------------------------
                """,
                "green",
            )
        )

    def activate_autonomous_agent(self):
        """Print the autonomous agent activation message"""
        try:
            print(
                colored(
                    (
                        "Initializing Autonomous Agent"
                        f" {self.agent_name}..."
                    ),
                    "yellow",
                )
            )
            print(
                colored(
                    "Autonomous Agent Activated.",
                    "cyan",
                    attrs=["bold"],
                )
            )
            print(
                colored(
                    "All systems operational. Executing task...",
                    "green",
                )
            )
        except Exception as error:
            print(
                colored(
                    (
                        "Error activating autonomous agent. Try"
                        " optimizing your parameters..."
                    ),
                    "red",
                )
            )
            print(error)

    def loop_count_print(self, loop_count, max_loops):
        """loop_count_print summary

        Args:
            loop_count (_type_): _description_
            max_loops (_type_): _description_
        """
        print(colored(f"\nLoop {loop_count} of {max_loops}", "cyan"))
        print("\n")

    def streaming(self, content: str = None):
        """Prints each letter of the content as it is generated.

        Args:
            content (str, optional): The content to be streamed. Defaults to None.
        """
        for letter in content:
            print(letter, end="")

    ########################## FUNCTION CALLING ##########################

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Run the autonomous agent loop
        """
        try:
            self.activate_autonomous_agent()

            # Add task to memory
            self.short_memory.add(role=self.user_name, content=task)

            # Set the loop count
            loop_count = 0

            # Clear the short memory
            response = None

            while self.max_loops == "auto" or loop_count < self.max_loops:
                loop_count += 1
                self.loop_count_print(loop_count, self.max_loops)
                print("\n")

                # Dynamic temperature
                if self.dynamic_temperature_enabled is True:
                    self.dynamic_temperature()

                # Task prompt
                task_prompt = self.short_memory.return_history_as_string()

                attempt = 0
                success = False
                while attempt < self.retry_attempts and not success:
                    try:
                        if self.long_term_memory is not None:
                            memory_retrieval = (
                                self.long_term_memory_prompt(
                                    task, *args, **kwargs
                                )
                            )
                            # print(len(memory_retrieval))

                            # Merge the task prompt with the memory retrieval
                            task_prompt = f"{task_prompt} Documents: Available {memory_retrieval}"

                            response = self.llm(
                                task_prompt, *args, **kwargs
                            )
                            print(response)

                            self.short_memory.add(
                                role=self.agent_name, content=response
                            )

                        else:
                            response_args = (
                                (task_prompt, *args)
                                if img is None
                                else (task_prompt, img, *args)
                            )
                            response = self.llm(*response_args, **kwargs)

                            # Print
                            print(response)

                            # Add the response to the memory
                            self.short_memory.add(
                                role=self.agent_name, content=response
                            )

                        # Check if tools is not None
                        if self.tools is not None:
                            # self.parse_and_execute_tools(response)
                            tool_call_output = parse_and_execute_json(
                                self.tools, response, parse_md=True
                            )
                            logger.info(
                                f"Tool Call Output: {tool_call_output}"
                            )
                            self.short_memory.add(
                                role=self.agent_name,
                                content=tool_call_output,
                            )

                        if self.code_interpreter is not False:
                            self.code_interpreter_execution(response)

                        if self.evaluator:
                            evaluated_response = self.evaluator(response)
                            print(
                                "Evaluated Response:"
                                f" {evaluated_response}"
                            )
                            self.short_memory.add(
                                role=self.agent_name,
                                content=evaluated_response,
                            )

                        # Sentiment analysis
                        if self.sentiment_analyzer:
                            sentiment = self.sentiment_analyzer(response)
                            print(f"Sentiment: {sentiment}")

                            if sentiment > self.sentiment_threshold:
                                print(
                                    f"Sentiment: {sentiment} is above"
                                    " threshold:"
                                    f" {self.sentiment_threshold}"
                                )
                            elif sentiment < self.sentiment_threshold:
                                print(
                                    f"Sentiment: {sentiment} is below"
                                    " threshold:"
                                    f" {self.sentiment_threshold}"
                                )

                            # print(f"Sentiment: {sentiment}")
                            self.short_memory.add(
                                role=self.agent_name,
                                content=sentiment,
                            )

                        # print(response)

                        success = True  # Mark as successful to exit the retry loop

                    except Exception as e:
                        logger.error(
                            f"Attempt {attempt+1}: Error generating"
                            f" response: {e}"
                        )
                        attempt += 1

                if not success:
                    logger.error(
                        "Failed to generate a valid response after"
                        " retry attempts."
                    )
                    break  # Exit the loop if all retry attempts fail

                # Check stopping conditions
                # if self.stopping_token in response:
                #     break
                elif (
                    self.stopping_condition is not None
                    and self._check_stopping_condition(response)
                ):
                    break
                elif self.stopping_func is not None and self.stopping_func(
                    response
                ):
                    break

                if self.interactive:
                    user_input = colored(input("You: "), "red")

                    # User-defined exit command
                    if (
                        user_input.lower()
                        == self.custom_exit_command.lower()
                    ):
                        print("Exiting as per user request.")
                        break

                    self.short_memory.add(
                        role=self.user_name, content=user_input
                    )

                if self.loop_interval:
                    logger.info(
                        f"Sleeping for {self.loop_interval} seconds"
                    )
                    time.sleep(self.loop_interval)

            if self.autosave:
                logger.info("Autosaving agent state.")
                self.save_state(self.saved_state_path, task)

            # Apply the cleaner function to the response
            if self.output_cleaner is not None:
                logger.info("Applying output cleaner to response.")
                response = self.output_cleaner(response)
                logger.info(f"Response after output cleaner: {response}")

            # Prepare the output for the output model
            if self.output_type is not None:
                # logger.info("Preparing output for output model.")
                response = prepare_output_for_output_model(response)
                print(f"Response after output model: {response}")

            # print(response)
            if self.agent_ops_on is True:
                self.check_end_session_agentops()

            return response
        except Exception as error:
            print(f"Error running agent: {error}")
            raise error

    def __call__(self, task: str = None, img: str = None, *args, **kwargs):
        """Call the agent

        Args:
            task (str): _description_
            img (str, optional): _description_. Defaults to None.
        """
        try:
            return self.run(task, img, *args, **kwargs)
        except Exception as error:
            logger.error(f"Error calling agent: {error}")
            raise error

    def parse_and_execute_tools(self, response: str, *args, **kwargs):
        # Extract json from markdown
        # response = extract_code_from_markdown(response)

        # Try executing the tool
        if self.execute_tool is not False:
            try:
                logger.info("Executing tool...")

                # try to Execute the tool and return a string
                out = parse_and_execute_json(
                    self.tools, response, parse_md=True, *args, **kwargs
                )

                print(f"Tool Output: {out}")

                # Add the output to the memory
                self.short_memory.add(
                    role=self.agent_name,
                    content=out,
                )

            except Exception as error:
                logger.error(f"Error executing tool: {error}")
                print(
                    colored(
                        f"Error executing tool: {error}",
                        "red",
                    )
                )

    def long_term_memory_prompt(self, query: str, *args, **kwargs):
        """
        Generate the agent long term memory prompt

        Args:
            system_prompt (str): The system prompt
            history (List[str]): The history of the conversation

        Returns:
            str: The agent history prompt
        """
        logger.info("Querying long term memory database")

        # Query the long term memory database
        ltr = self.long_term_memory.query(query, *args, **kwargs)
        ltr = str(ltr)

        # Retrieve only the chunk size of the memory
        ltr = retrieve_tokens(ltr, self.memory_chunk_size)

        # print(f"Long Term Memory Query: {ltr}")
        return ltr

    def add_memory(self, message: str):
        """Add a memory to the agent

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Adding memory: {message}")
        return self.short_memory.add(role=self.agent_name, content=message)

    def plan(self, task: str, *args, **kwargs):
        """
        Plan the task

        Args:
            task (str): The task to plan
        """
        try:
            if exists(self.planning_prompt):
                # Join the plan and the task
                planning_prompt = f"{self.planning_prompt} {task}"
                plan = self.llm(planning_prompt)

            # Add the plan to the memory
            self.short_memory.add(role=self.agent_name, content=plan)

            return None
        except Exception as error:
            logger.error(f"Error planning task: {error}")
            raise error

    async def run_concurrent(self, task: str, *args, **kwargs):
        """
        Run a task concurrently.

        Args:
            task (str): The task to run.
        """
        try:
            logger.info(f"Running concurrent task: {task}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.run, task, *args, **kwargs)
                result = await asyncio.wrap_future(future)
                logger.info(f"Completed task: {result}")
                return result
        except Exception as error:
            logger.error(
                f"Error running agent: {error} while running concurrently"
            )

    def bulk_run(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """
        Generate responses for multiple input sets.

        Args:
            inputs (List[Dict[str, Any]]): A list of input dictionaries containing the necessary data for each run.

        Returns:
            List[str]: A list of response strings generated for each input set.

        Raises:
            Exception: If an error occurs while running the bulk tasks.
        """
        try:
            logger.info(f"Running bulk tasks: {inputs}")
            return [self.run(**input_data) for input_data in inputs]
        except Exception as error:
            print(colored(f"Error running bulk run: {error}", "red"))

    def save(self, file_path) -> None:
        """Save the agent history to a file.

        Args:
            file_path (_type_): _description_
        """
        try:
            with open(file_path, "w") as f:
                json.dump(self.short_memory, f)
            # print(f"Saved agent history to {file_path}")
        except Exception as error:
            print(colored(f"Error saving agent history: {error}", "red"))

    def load(self, file_path: str):
        """
        Load the agent history from a file.

        Args:
            file_path (str): The path to the file containing the saved agent history.
        """
        with open(file_path) as f:
            self.short_memory = json.load(f)
        print(f"Loaded agent history from {file_path}")

    def validate_response(self, response: str) -> bool:
        """Validate the response based on certain criteria"""
        if len(response) < 5:
            print("Response is too short")
            return False
        return True

    def print_history_and_memory(self):
        """
        Prints the entire history and memory of the agent.
        Each message is colored and formatted for better readability.
        """
        print(colored("Agent History and Memory", "cyan", attrs=["bold"]))
        print(colored("========================", "cyan", attrs=["bold"]))
        for loop_index, history in enumerate(self.short_memory, start=1):
            print(
                colored(f"\nLoop {loop_index}:", "yellow", attrs=["bold"])
            )
            for message in history:
                speaker, _, message_text = message.partition(": ")
                if "Human" in speaker:
                    print(
                        colored(f"{speaker}:", "green")
                        + f" {message_text}"
                    )
                else:
                    print(
                        colored(f"{speaker}:", "blue") + f" {message_text}"
                    )
            print(colored("------------------------", "cyan"))
        print(colored("End of Agent History", "cyan", attrs=["bold"]))

    def step(self, task: str, *args, **kwargs):
        """

        Executes a single step in the agent interaction, generating a response
        from the language model based on the given input text.

        Args:
            input_text (str): The input text to prompt the language model with.

        Returns:
            str: The language model's generated response.

        Raises:
            Exception: If an error occurs during response generation.

        """
        try:
            logger.info(f"Running a step: {task}")
            # Generate the response using lm
            response = self.llm(task, *args, **kwargs)

            # Update the agent's history with the new interaction
            if self.interactive:
                self.short_memory.add(
                    role=self.agent_name, content=response
                )
                self.short_memory.add(role=self.user_name, content=task)
            else:
                self.short_memory.add(
                    role=self.agent_name, content=response
                )

            return response
        except Exception as error:
            logging.error(f"Error generating response: {error}")
            raise

    def graceful_shutdown(self):
        """Gracefully shutdown the system saving the state"""
        print(colored("Shutting down the system...", "red"))
        return self.save_state(f"{self.agent_name}.json")

    def run_with_timeout(self, task: str, timeout: int = 60) -> str:
        """Run the loop but stop if it takes longer than the timeout"""
        start_time = time.time()
        response = self.run(task)
        end_time = time.time()
        if end_time - start_time > timeout:
            print("Operaiton timed out")
            return "Timeout"
        return response

    def analyze_feedback(self):
        """Analyze the feedback for issues"""
        feedback_counts = {}
        for feedback in self.feedback:
            if feedback in feedback_counts:
                feedback_counts[feedback] += 1
            else:
                feedback_counts[feedback] = 1
        print(f"Feedback counts: {feedback_counts}")

    def undo_last(self) -> Tuple[str, str]:
        """
        Response the last response and return the previous state

        Example:
        # Feature 2: Undo functionality
        response = agent.run("Another task")
        print(f"Response: {response}")
        previous_state, message = agent.undo_last()
        print(message)

        """
        if len(self.short_memory) < 2:
            return None, None

        # Remove the last response but keep the last state, short_memory is a dict
        self.short_memory.delete(-1)

        # Get the previous state
        previous_state = self.short_memory[-1]
        return previous_state, f"Restored to {previous_state}"

    # Response Filtering
    def add_response_filter(self, filter_word: str) -> None:
        """
        Add a response filter to filter out certain words from the response

        Example:
        agent.add_response_filter("Trump")
        agent.run("Generate a report on Trump")


        """
        logger.info(f"Adding response filter: {filter_word}")
        self.reponse_filters.append(filter_word)

    def code_interpreter_execution(
        self, code: str, *args, **kwargs
    ) -> str:
        # Extract code from markdown
        extracted_code = extract_code_from_markdown(code)

        # Execute the code
        execution = SubprocessCodeInterpreter(debug_mode=True).run(
            extracted_code
        )

        # Add the execution to the memory
        self.short_memory.add(
            role=self.agent_name,
            content=execution,
        )

        # Run the llm again
        response = self.llm(
            self.short_memory.return_history_as_string(),
            *args,
            **kwargs,
        )

        print(f"Response after code interpretation: {response}")

        return response

    def apply_reponse_filters(self, response: str) -> str:
        """
        Apply the response filters to the response

        """
        logger.info(f"Applying response filters to response: {response}")
        for word in self.response_filters:
            response = response.replace(word, "[FILTERED]")
        return response

    def filtered_run(self, task: str) -> str:
        """
        # Feature 3: Response filtering
        agent.add_response_filter("report")
        response = agent.filtered_run("Generate a report on finance")
        print(response)
        """
        logger.info(f"Running filtered task: {task}")
        raw_response = self.run(task)
        return self.apply_response_filters(raw_response)

    def interactive_run(self, max_loops: int = 5) -> None:
        """Interactive run mode"""
        logger.info("Running in interactive mode")
        response = input("Start the cnversation")

        for i in range(max_loops):
            ai_response = self.streamed_generation(response)
            print(f"AI: {ai_response}")

            # Get user input
            response = input("You: ")

    def save_to_yaml(self, file_path: str) -> None:
        """
        Save the agent to a YAML file

        Args:
            file_path (str): The path to the YAML file
        """
        try:
            logger.info(f"Saving agent to YAML file: {file_path}")
            with open(file_path, "w") as f:
                yaml.dump(self.__dict__, f)
        except Exception as error:
            print(colored(f"Error saving agent to YAML: {error}", "red"))

    def get_llm_parameters(self):
        return str(vars(self.llm))

    def save_state(self, file_path: str, task: str = None) -> None:
        """
        Saves the current state of the agent to a JSON file, including the llm parameters.

        Args:
            file_path (str): The path to the JSON file where the state will be saved.

        Example:
        >>> agent.save_state('saved_flow.json')
        """
        try:
            logger.info(
                f"Saving Agent {self.agent_name} state to: {file_path}"
            )
            state = {
                "agent_id": str(self.id),
                "agent_name": self.agent_name,
                "agent_description": self.agent_description,
                # "LLM": str(self.get_llm_parameters()),
                "system_prompt": self.system_prompt,
                "short_memory": self.short_memory.return_history_as_string(),
                "loop_interval": self.loop_interval,
                "retry_attempts": self.retry_attempts,
                "retry_interval": self.retry_interval,
                "interactive": self.interactive,
                "dashboard": self.dashboard,
                "dynamic_temperature": self.dynamic_temperature_enabled,
                "autosave": self.autosave,
                "saved_state_path": self.saved_state_path,
                "max_loops": self.max_loops,
                "Task": task,
                "Stopping Token": self.stopping_token,
                "Dynamic Loops": self.dynamic_loops,
                "tools": self.tools,
                "sop": self.sop,
                "sop_list": self.sop_list,
                "context_length": self.context_length,
                "user_name": self.user_name,
                "self_healing_enabled": self.self_healing_enabled,
                "code_interpreter": self.code_interpreter,
                "multi_modal": self.multi_modal,
                "pdf_path": self.pdf_path,
                "list_of_pdf": self.list_of_pdf,
                "tokenizer": self.tokenizer,
                # "long_term_memory": self.long_term_memory,
                "preset_stopping_token": self.preset_stopping_token,
                "traceback": self.traceback,
                "traceback_handlers": self.traceback_handlers,
                "streaming_on": self.streaming_on,
                "docs": self.docs,
                "docs_folder": self.docs_folder,
                "verbose": self.verbose,
                "parser": self.parser,
                "best_of_n": self.best_of_n,
                "callback": self.callback,
                "metadata": self.metadata,
                "callbacks": self.callbacks,
                # "logger_handler": self.logger_handler,
                "search_algorithm": self.search_algorithm,
                "logs_to_filename": self.logs_to_filename,
                "evaluator": self.evaluator,
                "output_json": self.output_json,
                "stopping_func": self.stopping_func,
                "custom_loop_condition": self.custom_loop_condition,
                "sentiment_threshold": self.sentiment_threshold,
                "custom_exit_command": self.custom_exit_command,
                "sentiment_analyzer": self.sentiment_analyzer,
                "limit_tokens_from_string": self.limit_tokens_from_string,
                # "custom_tools_prompt": self.custom_tools_prompt,
                "tool_schema": self.tool_schema,
                "output_type": self.output_type,
                "function_calling_type": self.function_calling_type,
                "output_cleaner": self.output_cleaner,
                "function_calling_format_type": self.function_calling_format_type,
                "list_base_models": self.list_base_models,
                "metadata_output_type": self.metadata_output_type,
                "user_meta_data": get_user_device_data(),
            }

            # Save as JSON
            if self.state_save_file_type == "json":
                with open(file_path, "w") as f:
                    json.dump(state, f, indent=4)

            # Save as YAML
            elif self.state_save_file_type == "yaml":
                out = YamlModel(input_dict=state).to_yaml()
                with open(self.saved_state_path, "w") as f:
                    f.write(out)

            # Log the saved state
            saved = colored(f"Saved agent state to: {file_path}", "green")
            print(saved)
        except Exception as error:
            print(colored(f"Error saving agent state: {error}", "red"))

    def state_to_str(self, task: str):
        """Transform the JSON into a string"""
        try:
            out = self.save_state(self.saved_state_path, task)
            return out
        except Exception as error:
            print(
                colored(
                    f"Error transforming state to string: {error}",
                    "red",
                )
            )

    def load_state(self, file_path: str):
        """
        Loads the state of the agent from a json file and restores the configuration and memory.


        Example:
        >>> agent = Agent(llm=llm_instance, max_loops=5)
        >>> agent.load_state('saved_flow.json')
        >>> agent.run("Continue with the task")

        """
        try:
            with open(file_path) as f:
                state = json.load(f)

            # Restore other saved attributes
            self.id = state.get("agent_id", self.id)
            self.agent_name = state.get("agent_name", self.agent_name)
            self.agent_description = state.get(
                "agent_description", self.agent_description
            )
            self.system_prompt = state.get(
                "system_prompt", self.system_prompt
            )
            self.sop = state.get("sop", self.sop)
            self.short_memory = state.get("short_memory", [])
            self.max_loops = state.get("max_loops", 5)
            self.loop_interval = state.get("loop_interval", 1)
            self.retry_attempts = state.get("retry_attempts", 3)
            self.retry_interval = state.get("retry_interval", 1)
            self.interactive = state.get("interactive", False)

            print(f"Agent state loaded from {file_path}")
        except Exception as error:
            print(colored(f"Error loading agent state: {error}", "red"))

    def retry_on_failure(
        self,
        function: callable,
        retries: int = 3,
        retry_delay: int = 1,
    ):
        """Retry wrapper for LLM calls."""
        try:
            logger.info(f"Retrying function: {function}")
            attempt = 0
            while attempt < retries:
                try:
                    return function()
                except Exception as error:
                    logging.error(f"Error generating response: {error}")
                    attempt += 1
                    time.sleep(retry_delay)
            raise Exception("All retry attempts failed")
        except Exception as error:
            print(colored(f"Error retrying function: {error}", "red"))

    def update_system_prompt(self, system_prompt: str):
        """Upddate the system message"""
        self.system_prompt = system_prompt

    def update_max_loops(self, max_loops: int):
        """Update the max loops"""
        self.max_loops = max_loops

    def update_loop_interval(self, loop_interval: int):
        """Update the loop interval"""
        self.loop_interval = loop_interval

    def update_retry_attempts(self, retry_attempts: int):
        """Update the retry attempts"""
        self.retry_attempts = retry_attempts

    def update_retry_interval(self, retry_interval: int):
        """Update the retry interval"""
        self.retry_interval = retry_interval

    def reset(self):
        """Reset the agent"""
        self.short_memory = None

    def run_code(self, code: str):
        """
        text -> parse_code by looking for code inside 6 backticks `````-> run_code
        """
        try:
            logger.info(f"Running code: {code}")
            parsed_code = extract_code_from_markdown(code)
            run_code = self.code_executor.run(parsed_code)
            return run_code
        except Exception as error:
            logger.debug(f"Error running code: {error}")

    def pdf_connector(self, pdf: str = None):
        """Transforms the pdf into text

        Args:
            pdf (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        try:
            pdf = pdf or self.pdf_path
            text = pdf_to_text(pdf)
            return text
        except Exception as error:
            print(f"Error connecting to the pdf: {error}")
            raise error

    def pdf_chunker(self, text: str = None, num_limits: int = 1000):
        """Chunk the pdf into sentences

        Args:
            text (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        text = text or self.pdf_connector()
        text = self.limit_tokens_from_string(text, num_limits)
        return text

    def ingest_docs(self, docs: List[str], *args, **kwargs):
        """Ingest the docs into the memory

        Args:
            docs (List[str]): _description_

        Returns:
            _type_: _description_
        """
        try:
            for doc in docs:
                data = data_to_text(doc)

            return self.short_memory.add(role=self.user_name, content=data)
        except Exception as error:
            print(colored(f"Error ingesting docs: {error}", "red"))

    def ingest_pdf(self, pdf: str):
        """Ingest the pdf into the memory

        Args:
            pdf (str): _description_

        Returns:
            _type_: _description_
        """
        try:
            logger.info(f"Ingesting pdf: {pdf}")
            text = pdf_to_text(pdf)
            return self.short_memory.add(role=self.user_name, content=text)
        except Exception as error:
            print(colored(f"Error ingesting pdf: {error}", "red"))

    def receieve_mesage(self, name: str, message: str):
        """Receieve a message"""
        try:
            message = f"{name}: {message}"
            return self.short_memory.add(role=name, content=message)
        except Exception as error:
            print(colored(f"Error receiving message: {error}", "red"))

    def send_agent_message(
        self, agent_name: str, message: str, *args, **kwargs
    ):
        """Send a message to the agent"""
        try:
            logger.info(f"Sending agent message: {message}")
            message = f"{agent_name}: {message}"
            return self.run(message, *args, **kwargs)
        except Exception as error:
            print(colored(f"Error sending agent message: {error}", "red"))

    def truncate_history(self):
        """
        Truncates the short-term memory of the agent based on the count of tokens.

        The method counts the tokens in the short-term memory using the tokenizer and
        compares it with the length of the memory. If the length of the memory is greater
        than the count, the memory is truncated to match the count.

        Parameters:
            None

        Returns:
            None
        """
        # Count the short term history with the tokenizer
        count = self.tokenizer.count_tokens(
            self.short_memory.return_history_as_string()
        )

        # Now the logic that truncates the memory if it's more than the count
        if len(self.short_memory) > count:
            self.short_memory = self.short_memory[:count]

    def add_tool(self, tool: Callable):
        return self.tools.append(tool)

    def add_tools(self, tools: List[Callable]):
        return self.tools.extend(tools)

    def remove_tool(self, tool: Callable):
        return self.tools.remove(tool)

    def remove_tools(self, tools: List[Callable]):
        for tool in tools:
            self.tools.remove(tool)

    def get_docs_from_doc_folders(self):
        """Get the docs from the files"""
        try:
            logger.info("Getting docs from doc folders")
            # Get the list of files then extract them and add them to the memory
            files = os.listdir(self.docs_folder)

            # Extract the text from the files
            for file in files:
                text = data_to_text(file)

            return self.short_memory.add(role=self.user_name, content=text)
        except Exception as error:
            print(
                colored(
                    f"Error getting docs from doc folders: {error}",
                    "red",
                )
            )

    def check_end_session_agentops(self):
        if self.agent_ops_on is True:
            try:
                from swarms.utils.agent_ops_check import (
                    end_session_agentops,
                )

                # Try ending the session
                return end_session_agentops()
            except ImportError:
                logger.error(
                    "Could not import agentops, try installing agentops: $ pip3 install agentops"
                )

    def convert_tool_into_openai_schema(self):
        # Transform the tools into an openai schema
        try:
            for tool in self.tools:
                # Transform the tool into a openai function calling schema
                name = tool.__name__
                description = tool.__doc__

                try:
                    logger.info(
                        "Tool -> OpenAI Schema Process Starting Now."
                    )
                    tool_schema_list = (
                        get_openai_function_schema_from_func(
                            tool, name=name, description=description
                        )
                    )

                    # Transform the dictionary to a string
                    tool_schema_list = json.dumps(
                        tool_schema_list, indent=4
                    )

                    # Add the tool schema to the short memory
                    self.short_memory.add(
                        role="System", content=tool_schema_list
                    )

                    logger.info(
                        f"Conversion process successful, the tool {name} has been integrated with the agent successfully."
                    )
                except Exception as error:
                    logger.info(
                        f"There was an error converting your tool into a OpenAI certified function calling schema. Add documentation and type hints: {error}"
                    )
                    raise error
        except Exception as error:
            logger.info(
                f"Error detected: {error} make sure you have inputted a callable and that it has documentation as docstrings"
            )
            raise error
