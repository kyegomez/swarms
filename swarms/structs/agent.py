import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from loguru import logger
from termcolor import colored

from swarms.memory.base_vectordb import AbstractVectorDatabase
from swarms.prompts.agent_system_prompts import AGENT_SYSTEM_PROMPT_3
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)
from swarms.prompts.worker_prompt import worker_tools_sop_promp
from swarms.structs.conversation import Conversation
from swarms.structs.schemas import Step
from swarms.tokenizers.base_tokenizer import BaseTokenizer
from swarms.tools.exec_tool import execute_tool_by_name
from swarms.tools.tool import BaseTool
from swarms.utils.code_interpreter import SubprocessCodeInterpreter
from swarms.utils.data_to_text import data_to_text
from swarms.utils.parse_code import extract_code_from_markdown
from swarms.utils.pdf_to_text import pdf_to_text
from swarms.utils.token_count_tiktoken import limit_tokens_from_string
from swarms.utils.video_to_frames import (
    save_frames_as_images,
    video_to_frames,
)


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


# Step ID generator
def step_id():
    return str(uuid.uuid1())


class Agent:
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
        memory (AbstractVectorDatabase): The memory
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
        id: str = agent_id,
        llm: Any = None,
        template: Optional[str] = None,
        max_loops: Optional[int] = 1,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: int = 1,
        retry_attempts: int = 3,
        retry_interval: int = 1,
        return_history: bool = False,
        stopping_token: str = None,
        dynamic_loops: Optional[bool] = False,
        interactive: bool = False,
        dashboard: bool = False,
        agent_name: str = "swarm-worker-01",
        agent_description: str = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT_3,
        tools: List[BaseTool] = None,
        dynamic_temperature_enabled: Optional[bool] = False,
        sop: Optional[str] = None,
        sop_list: Optional[List[str]] = None,
        saved_state_path: Optional[str] = None,
        autosave: Optional[bool] = False,
        context_length: Optional[int] = 8192,
        user_name: str = "Human:",
        self_healing_enabled: Optional[bool] = False,
        code_interpreter: Optional[bool] = False,
        multi_modal: Optional[bool] = None,
        pdf_path: Optional[str] = None,
        list_of_pdf: Optional[str] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        long_term_memory: Optional[AbstractVectorDatabase] = None,
        preset_stopping_token: Optional[bool] = False,
        traceback: Any = None,
        traceback_handlers: Any = None,
        streaming_on: Optional[bool] = False,
        docs: List[str] = None,
        docs_folder: str = None,
        verbose: bool = False,
        parser: Optional[Callable] = None,
        best_of_n: Optional[int] = None,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callable]] = None,
        logger_handler: Any = sys.stderr,
        search_algorithm: Optional[Callable] = None,
        logs_to_filename: Optional[str] = None,
        evaluator: Optional[Callable] = None,
        output_json: bool = False,
        stopping_func: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
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

        # The max_loops will be set dynamically if the dynamic_loop
        if self.dynamic_loops:
            logger.info("Dynamic loops enabled")
            self.max_loops = "auto"

        # If multimodal = yes then set the sop to the multimodal sop
        if self.multi_modal:
            self.sop = MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1

        # If the user inputs a list of strings for the sop then join them and set the sop
        if self.sop_list:
            self.sop = "\n".join(self.sop_list)

        # Memory
        self.feedback = []

        # Initialize the code executor
        self.code_executor = SubprocessCodeInterpreter(
            debug_mode=True,
        )

        # If the preset stopping token is enabled then set the stopping token to the preset stopping token
        if preset_stopping_token:
            self.stopping_token = "<DONE>"

        self.short_memory = Conversation(
            system_prompt=self.system_prompt, time_enabled=True
        )

        # If the docs exist then ingest the docs
        if self.docs:
            self.ingest_docs(self.docs)

        # If docs folder exists then get the docs from docs folder
        if self.docs_folder:
            self.get_docs_from_doc_folders()

        # If tokenizer and context length exists then:
        # if self.tokenizer and self.context_length:
        #     self.truncate_history()

        # If verbose is enabled then set the logger level to info
        # if verbose:
        #     logger.setLevel(logging.INFO)

        # If tools are provided then set the tool prompt by adding to sop
        if self.tools:
            tools_prompt = worker_tools_sop_promp(
                name=self.agent_name,
                memory=self.short_memory.return_history_as_string(),
            )

            # Append the tools prompt to the sop
            self.sop = f"{self.sop}\n{tools_prompt}"

        # If the long term memory is provided then set the long term memory prompt

        # Agentic stuff
        self.reply = ""
        self.question = None
        self.answer = ""

        # Initialize the llm with the conditional variables
        # self.llm = llm(*args, **kwargs)

        # Step cache
        self.step_cache = []

        # Set the logger handler
        if logger_handler:
            logger.add(
                f"{self.agent_name}.log",
                level="INFO",
                colorize=True,
                format=(
                    "<green>{time}</green> <level>{message}</level>"
                ),
                backtrace=True,
                diagnose=True,
            )

        # logger.info("Creating Agent {}".format(self.agent_name))

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
                colored(
                    f"Error dynamically changing temperature: {error}"
                )
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
            print(
                colored(
                    f"Error adding task to memory: {error}", "red"
                )
            )

    def add_message_to_memory(self, message: str):
        """Add the message to the memory"""
        try:
            logger.info(f"Adding message to memory: {message}")
            self.short_memory.add(
                role=self.agent_name, content=message
            )
        except Exception as error:
            print(
                colored(
                    f"Error adding message to memory: {error}", "red"
                )
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
                    "Initializing Autonomous Agent"
                    f" {self.agent_name}...",
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
                    "Error activating autonomous agent. Try"
                    " optimizing your parameters...",
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
        """prints each chunk of content as it is generated

        Args:
            content (str, optional): _description_. Defaults to None.
        """
        for chunk in content:
            print(chunk, end="")

    def _history(self, user_name: str, task: str) -> str:
        """Generate the history for the history prompt

        Args:
            user_name (str): _description_
            task (str): _description_

        Returns:
            str: _description_
        """
        history = [f"{user_name}: {task}"]
        return history

    def _dynamic_prompt_setup(
        self, dynamic_prompt: str, task: str
    ) -> str:
        """_dynamic_prompt_setup summary

        Args:
            dynamic_prompt (str): _description_
            task (str): _description_

        Returns:
            str: _description_
        """
        dynamic_prompt = (
            dynamic_prompt or self.construct_dynamic_prompt()
        )
        combined_prompt = f"{dynamic_prompt}\n{task}"
        return combined_prompt

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        video: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Run the autonomous agent loop

        Args:
            task (str): The initial task to run

        Agent:
        1. Generate a response
        2. Check stopping condition
        3. If stopping condition is met, stop
        4. If stopping condition is not met, generate a response
        5. Repeat until stopping condition is met or max_loops is reached

        """
        try:
            if video:
                video_to_frames(video)
                frames = save_frames_as_images(video)
                for frame in frames:
                    img = frame

            # Activate Autonomous agent message
            self.activate_autonomous_agent()

            # response = task  # or combined_prompt
            history = self._history(self.user_name, task)

            # If dashboard = True then print the dashboard
            if self.dashboard:
                self.print_dashboard(task)

            loop_count = 0

            # While the max_loops is auto or the loop count is less than the max_loops
            while (
                self.max_loops == "auto"
                or loop_count < self.max_loops
            ):
                # Loop count
                loop_count += 1
                self.loop_count_print(loop_count, self.max_loops)
                print("\n")

                # Adjust temperature, comment if no work
                if self.dynamic_temperature_enabled:
                    print(colored("Adjusting temperature...", "blue"))
                    self.dynamic_temperature()

                # Preparing the prompt
                task = self.agent_history_prompt(history=task)

                attempt = 0
                while attempt < self.retry_attempts:
                    try:
                        if img:
                            response = self.llm(
                                task,
                                img,
                                **kwargs,
                            )
                            print(response)
                        else:
                            response = self.llm(
                                task,
                                **kwargs,
                            )
                            print(response)

                        if self.output_json:
                            response = extract_code_from_markdown(
                                response
                            )

                        # Add the response to the history
                        history.append(response)

                        # Log each step
                        step = Step(
                            input=str(task),
                            task_id=str(task_id),
                            step_id=str(step_id),
                            output=str(response),
                            status="running",
                        )

                        if self.evaluator:
                            evaluated_response = self.evaluator(
                                response
                            )

                            out = (
                                f"Response: {response}\nEvaluated"
                                f" Response: {evaluated_response}"
                            )
                            out = self.short_memory.add(
                                "Evaluator", out
                            )

                        # Stopping logic for agents
                        if self.stopping_token:
                            # Check if the stopping token is in the response
                            if self.stopping_token in response:
                                break

                        if self.stopping_condition:
                            if self._check_stopping_condition(
                                response
                            ):
                                break

                        # if self.parse_done_token:
                        #     if parse_done_token(response):
                        #         break

                        if self.stopping_func is not None:
                            if self.stopping_func(response) is True:
                                break

                        # If the stopping condition is met then break
                        self.step_cache.append(step)
                        logging.info(f"Step: {step}")

                        # If parser exists then parse the response
                        if self.parser:
                            response = self.parser(response)

                        # If code interpreter is enabled then run the code
                        if self.code_interpreter:
                            self.run_code(response)

                        # If tools are enabled then execute the tools
                        if self.tools:
                            execute_tool_by_name(
                                response,
                                self.tools,
                                self.stopping_condition,
                            )

                        # If interactive mode is enabled then print the response and get user input
                        if self.interactive:
                            print(f"AI: {response}")
                            history.append(f"AI: {response}")
                            response = input("You: ")
                            history.append(f"Human: {response}")

                        # If interactive mode is not enabled then print the response
                        else:
                            # print(f"AI: {response}")
                            history.append(f"AI: {response}")
                            # print(response)
                        break
                    except Exception as e:
                        logging.error(
                            f"Error generating response: {e}"
                        )
                        attempt += 1
                        time.sleep(self.retry_interval)

                time.sleep(self.loop_interval)
            # Add the history to the memory
            self.short_memory.add(
                role=self.agent_name, content=history
            )

            # If autosave is enabled then save the state
            if self.autosave:
                print(
                    colored(
                        "Autosaving agent state to"
                        f" {self.saved_state_path}",
                        "green",
                    )
                )
                self.save_state(self.saved_state_path)

            # If return history is enabled then return the response and history
            if self.return_history:
                return response, history

            return response
        except Exception as error:
            logger.error(f"Error running agent: {error}")
            raise

    def __call__(self, task: str, img: str = None, *args, **kwargs):
        """Call the agent

        Args:
            task (str): _description_
            img (str, optional): _description_. Defaults to None.
        """
        self.run(task, img, *args, **kwargs)

    def agent_history_prompt(
        self,
        history: str = None,
    ):
        """
        Generate the agent history prompt

        Args:
            system_prompt (str): The system prompt
            history (List[str]): The history of the conversation

        Returns:
            str: The agent history prompt
        """
        if self.sop:
            system_prompt = self.system_prompt
            agent_history_prompt = f"""
                role: system
                {system_prompt}

                Follow this standard operating procedure (SOP) to complete tasks:
                {self.sop}
                
                {history}
            """
            return agent_history_prompt
        else:
            system_prompt = self.system_prompt
            agent_history_prompt = f"""
                System : {system_prompt}
                
                {history}
            """
            return agent_history_prompt

    def long_term_memory_prompt(self, query: str, *args, **kwargs):
        """
        Generate the agent long term memory prompt

        Args:
            system_prompt (str): The system prompt
            history (List[str]): The history of the conversation

        Returns:
            str: The agent history prompt
        """
        ltr = str(self.long_term_memory.query(query), *args, **kwargs)

        context = f"""
            System: This reminds you of these events from your past: [{ltr}]
        """
        return self.short_memory.add(
            role=self.agent_name, content=context
        )

    def add_memory(self, message: str):
        """Add a memory to the agent

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Adding memory: {message}")
        return self.short_memory.add(
            role=self.agent_name, content=message
        )

    async def run_concurrent(self, tasks: List[str], **kwargs):
        """
        Run a batch of tasks concurrently and handle an infinite level of task inputs.

        Args:
            tasks (List[str]): A list of tasks to run.
        """
        try:
            logger.info(f"Running concurrent tasks: {tasks}")
            task_coroutines = [
                self.run_async(task, **kwargs) for task in tasks
            ]
            completed_tasks = await asyncio.gather(*task_coroutines)
            logger.info(f"Completed tasks: {completed_tasks}")
            return completed_tasks
        except Exception as error:
            print(
                colored(
                    f"Error running agent: {error} while running"
                    " concurrently",
                    "red",
                )
            )

    def bulk_run(self, inputs: List[Dict[str, Any]]) -> List[str]:
        try:
            """Generate responses for multiple input sets."""
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
            print(
                colored(f"Error saving agent history: {error}", "red")
            )

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
        print(
            colored(
                "Agent History and Memory", "cyan", attrs=["bold"]
            )
        )
        print(
            colored(
                "========================", "cyan", attrs=["bold"]
            )
        )
        for loop_index, history in enumerate(
            self.short_memory, start=1
        ):
            print(
                colored(
                    f"\nLoop {loop_index}:", "yellow", attrs=["bold"]
                )
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
                        colored(f"{speaker}:", "blue")
                        + f" {message_text}"
                    )
            print(colored("------------------------", "cyan"))
        print(colored("End of Agent History", "cyan", attrs=["bold"]))

    def step(self, task: str, **kwargs):
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
            logger.info(f"Running a single step: {task}")
            # Generate the response using lm
            response = self.llm(task, **kwargs)

            # Update the agent's history with the new interaction
            if self.interactive:
                self.short_memory.add(
                    role=self.agent_name, content=response
                )
                self.short_memory.add(
                    role=self.user_name, content=task
                )
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
        return self.save_state("flow_state.json")

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

    def apply_reponse_filters(self, response: str) -> str:
        """
        Apply the response filters to the response

        """
        logger.info(
            f"Applying response filters to response: {response}"
        )
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
            print(
                colored(f"Error saving agent to YAML: {error}", "red")
            )

    def save_state(self, file_path: str) -> None:
        """
        Saves the current state of the agent to a JSON file, including the llm parameters.

        Args:
            file_path (str): The path to the JSON file where the state will be saved.

        Example:
        >>> agent.save_state('saved_flow.json')
        """
        try:
            logger.info(f"Saving agent state to: {file_path}")
            state = {
                "agent_id": str(self.id),
                "agent_name": self.agent_name,
                "agent_description": self.agent_description,
                "system_prompt": self.system_prompt,
                "sop": self.sop,
                "short_memory": (
                    self.short_memory.return_history_as_string()
                ),
                "loop_interval": self.loop_interval,
                "retry_attempts": self.retry_attempts,
                "retry_interval": self.retry_interval,
                "interactive": self.interactive,
                "dashboard": self.dashboard,
                "dynamic_temperature": (
                    self.dynamic_temperature_enabled
                ),
                "autosave": self.autosave,
                "saved_state_path": self.saved_state_path,
                "max_loops": self.max_loops,
            }

            with open(file_path, "w") as f:
                json.dump(state, f, indent=4)

            saved = colored(
                f"Saved agent state to: {file_path}", "green"
            )
            print(saved)
        except Exception as error:
            print(
                colored(f"Error saving agent state: {error}", "red")
            )

    def state_to_str(self):
        """Transform the JSON into a string"""
        try:
            state = {
                "agent_id": str(self.id),
                "agent_name": self.agent_name,
                "agent_description": self.agent_description,
                "system_prompt": self.system_prompt,
                "sop": self.sop,
                "short_memory": (
                    self.short_memory.return_history_as_string()
                ),
                "loop_interval": self.loop_interval,
                "retry_attempts": self.retry_attempts,
                "retry_interval": self.retry_interval,
                "interactive": self.interactive,
                "dashboard": self.dashboard,
                "dynamic_temperature": (
                    self.dynamic_temperature_enabled
                ),
                "autosave": self.autosave,
                "saved_state_path": self.saved_state_path,
                "max_loops": self.max_loops,
            }
            out = str(state)
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
            print(
                colored(f"Error loading agent state: {error}", "red")
            )

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
                    logging.error(
                        f"Error generating response: {error}"
                    )
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
        self.short_memory = {}

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
        text = limit_tokens_from_string(text, num_limits)
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

            return self.short_memory.add(
                role=self.user_name, content=data
            )
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
            return self.short_memory.add(
                role=self.user_name, content=text
            )
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
            print(
                colored(
                    f"Error sending agent message: {error}", "red"
                )
            )

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

    def get_docs_from_doc_folders(self):
        """Get the docs from the files"""
        try:
            logger.info("Getting docs from doc folders")
            # Get the list of files then extract them and add them to the memory
            files = os.listdir(self.docs_folder)

            # Extract the text from the files
            for file in files:
                text = data_to_text(file)

            return self.short_memory.add(
                role=self.user_name, content=text
            )
        except Exception as error:
            print(
                colored(
                    f"Error getting docs from doc folders: {error}",
                    "red",
                )
            )
