import asyncio
import inspect
import json
import logging
import random
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from termcolor import colored

from swarms.memory.base_vector_db import VectorDatabase
from swarms.prompts.agent_system_prompts import (
    FLOW_SYSTEM_PROMPT,
    agent_system_prompt_2,
)
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)
from swarms.prompts.tools import (
    SCENARIOS,
)
from swarms.tools.tool import BaseTool
from swarms.utils.code_interpreter import SubprocessCodeInterpreter
from swarms.utils.parse_code import (
    extract_code_in_backticks_in_string,
)
from swarms.utils.pdf_to_text import pdf_to_text
from swarms.utils.token_count_tiktoken import limit_tokens_from_string


# Utils
# Custom stopping condition
def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "Stop" in response.lower()


# Parse done token
def parse_done_token(response: str) -> bool:
    """Parse the response to see if the done token is present"""
    return "<DONE>" in response


# Agent ID generator
def agent_id():
    """Generate an agent id"""
    return str(uuid.uuid4())


class Agent:
    """
    Agent is the structure that provides autonomy to any llm in a reliable and effective fashion.
    The agent structure is designed to be used with any llm and provides the following features:

    Features:
    * Interactive, AI generates, then user input
    * Message history and performance history fed -> into context -> truncate if too long
    * Ability to save and load flows
    * Ability to provide feedback on responses
    * Ability to provide a loop interval

    Args:
        id (str): The id of the agent
        llm (Any): The language model to use
        template (Optional[str]): The template to use
        max_loops (int): The maximum number of loops
        stopping_condition (Optional[Callable[[str], bool]]): The stopping condition
        loop_interval (int): The loop interval
        retry_attempts (int): The retry attempts
        retry_interval (int): The retry interval
        return_history (bool): Return the history
        stopping_token (str): The stopping token
        dynamic_loops (Optional[bool]): Dynamic loops
        interactive (bool): Interactive mode
        dashboard (bool): Dashboard mode
        agent_name (str): The name of the agent
        agent_description (str): The description of the agent
        system_prompt (str): The system prompt
        tools (List[BaseTool]): The tools
        dynamic_temperature_enabled (Optional[bool]): Dynamic temperature enabled
        sop (Optional[str]): The standard operating procedure
        sop_list (Optional[List[str]]): The standard operating procedure list
        saved_state_path (Optional[str]): The saved state path
        autosave (Optional[bool]): Autosave
        context_length (Optional[int]): The context length
        user_name (str): The user name
        self_healing_enabled (Optional[bool]): Self healing enabled
        code_interpreter (Optional[bool]): Code interpreter
        multi_modal (Optional[bool]): Multi modal
        pdf_path (Optional[str]): The pdf path
        list_of_pdf (Optional[str]): The list of pdf
        tokenizer (Optional[Any]): The tokenizer
        memory (Optional[VectorDatabase]): The memory
        preset_stopping_token (Optional[bool]): Preset stopping token
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Methods:
        run(task: str, **kwargs: Any): Run the agent on a task
        run_concurrent(tasks: List[str], **kwargs: Any): Run the agent on a list of tasks concurrently
        bulk_run(inputs: List[Dict[str, Any]]): Run the agent on a list of inputs
        from_llm_and_template(llm: Any, template: str): Create AgentStream from LLM and a string template.
        from_llm_and_template_file(llm: Any, template_file: str): Create AgentStream from LLM and a template file.
        save(file_path): Save the agent history to a file
        load(file_path): Load the agent history from a file
        validate_response(response: str): Validate the response based on certain criteria
        print_history_and_memory(): Print the entire history and memory of the agent
        step(task: str, **kwargs): Executes a single step in the agent interaction, generating a response from the language model based on the given input text.
        graceful_shutdown(): Gracefully shutdown the system saving the state
        run_with_timeout(task: str, timeout: int): Run the loop but stop if it takes longer than the timeout
        analyze_feedback(): Analyze the feedback for issues
        undo_last(): Response the last response and return the previous state
        add_response_filter(filter_word: str): Add a response filter to filter out certain words from the response
        apply_reponse_filters(response: str): Apply the response filters to the response
        filtered_run(task: str): Filtered run
        interactive_run(max_loops: int): Interactive run mode
        streamed_generation(prompt: str): Stream the generation of the response
        get_llm_params(): Extracts and returns the parameters of the llm object for serialization.
        save_state(file_path: str): Saves the current state of the agent to a JSON file, including the llm parameters.
        load_state(file_path: str): Loads the state of the agent from a json file and restores the configuration and memory.
        retry_on_failure(function, retries: int = 3, retry_delay: int = 1): Retry wrapper for LLM calls.
        run_code(response: str): Run the code in the response
        construct_dynamic_prompt(): Construct the dynamic prompt
        extract_tool_commands(text: str): Extract the tool commands from the text
        parse_and_execute_tools(response: str): Parse and execute the tools
        execute_tools(tool_name, params): Execute the tool with the provided params
        truncate_history(): Take the history and truncate it to fit into the model context length
        add_task_to_memory(task: str): Add the task to the memory
        add_message_to_memory(message: str): Add the message to the memory
        add_message_to_memory_and_truncate(message: str): Add the message to the memory and truncate
        print_dashboard(task: str): Print dashboard
        activate_autonomous_agent(): Print the autonomous agent activation message
        dynamic_temperature(): Dynamically change the temperature
        _check_stopping_condition(response: str): Check if the stopping condition is met
        format_prompt(template, **kwargs: Any): Format the template with the provided kwargs using f-string interpolation.
        get_llm_init_params(): Get LLM init params
        get_tool_description(): Get the tool description
        find_tool_by_name(name: str): Find a tool by name


    Example:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import Agent
    >>> llm = OpenAIChat(
    ...     openai_api_key=api_key,
    ...     temperature=0.5,
    ... )
    >>> agent = Agent(
    ...     llm=llm, max_loops=5,
    ...     #system_prompt=SYSTEM_PROMPT,
    ...     #retry_interval=1,
    ... )
    >>> agent.run("Generate a 10,000 word blog")
    >>> agent.save("path/agent.yaml")
    """

    def __init__(
        self,
        id: str = agent_id,
        llm: Any = None,
        template: Optional[str] = None,
        max_loops=5,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: int = 1,
        retry_attempts: int = 3,
        retry_interval: int = 1,
        return_history: bool = False,
        stopping_token: str = None,
        dynamic_loops: Optional[bool] = False,
        interactive: bool = False,
        dashboard: bool = False,
        agent_name: str = "Autonomous-Agent-XYZ1B",
        agent_description: str = None,
        system_prompt: str = FLOW_SYSTEM_PROMPT,
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
        tokenizer: Optional[Any] = None,
        memory: Optional[VectorDatabase] = None,
        preset_stopping_token: Optional[bool] = False,
        *args,
        **kwargs: Any,
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
        self.stopping_token = stopping_token  # or "<DONE>"
        self.interactive = interactive
        self.dashboard = dashboard
        self.return_history = return_history
        self.dynamic_temperature_enabled = dynamic_temperature_enabled
        self.dynamic_loops = dynamic_loops
        self.user_name = user_name
        self.context_length = context_length
        self.sop = sop
        self.sop_list = sop_list
        self.tools = tools or []
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
        self.memory = memory
        self.preset_stopping_token = preset_stopping_token

        # The max_loops will be set dynamically if the dynamic_loop
        if self.dynamic_loops:
            self.max_loops = "auto"

        # If multimodal = yes then set the sop to the multimodal sop
        if self.multi_modal:
            self.sop = MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1

        # If the user inputs a list of strings for the sop then join them and set the sop
        if self.sop_list:
            self.sop = "\n".join(self.sop_list)

        # Memory
        self.feedback = []
        self.short_memory = []

        # Initialize the code executor
        self.code_executor = SubprocessCodeInterpreter()

        # If the preset stopping token is enabled then set the stopping token to the preset stopping token
        if preset_stopping_token:
            self.stopping_token = "<DONE>"

    def provide_feedback(self, feedback: str) -> None:
        """Allow users to provide feedback on the responses."""
        self.feedback.append(feedback)
        logging.info(f"Feedback received: {feedback}")

    def _check_stopping_condition(self, response: str) -> bool:
        """Check if the stopping condition is met."""
        if self.stopping_condition:
            return self.stopping_condition(response)
        return False

    def dynamic_temperature(self):
        """
        1. Check the self.llm object for the temperature
        2. If the temperature is not present, then use the default temperature
        3. If the temperature is present, then dynamically change the temperature
        4. for every loop you can randomly change the temperature on a scale from 0.0 to 1.0
        """
        if hasattr(self.llm, "temperature"):
            # Randomly change the temperature attribute of self.llm object
            self.llm.temperature = random.uniform(0.0, 1.0)
        else:
            # Use a default temperature
            self.llm.temperature = 0.7

    def format_prompt(self, template, **kwargs: Any) -> str:
        """Format the template with the provided kwargs using f-string interpolation."""
        return template.format(**kwargs)

    def get_llm_init_params(self) -> str:
        """Get LLM init params"""
        init_signature = inspect.signature(self.llm.__init__)
        params = init_signature.parameters
        params_str_list = []

        for name, param in params.items():
            if name == "self":
                continue
            if hasattr(self.llm, name):
                value = getattr(self.llm, name)
            else:
                value = self.llm.__dict__.get(name, "Unknown")

            params_str_list.append(
                f"    {name.capitalize().replace('_', ' ')}: {value}"
            )

        return "\n".join(params_str_list)

    def get_tool_description(self):
        """Get the tool description"""
        if self.tools:
            try:
                tool_descriptions = []
                for tool in self.tools:
                    description = f"{tool.name}: {tool.description}"
                    tool_descriptions.append(description)
                return "\n".join(tool_descriptions)
            except Exception as error:
                print(
                    f"Error getting tool description: {error} try"
                    " adding a description to the tool or removing"
                    " the tool"
                )
        else:
            return "No tools available"

    def find_tool_by_name(self, name: str):
        """Find a tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def construct_dynamic_prompt(self):
        """Construct the dynamic prompt"""
        tools_description = self.get_tool_description()

        tool_prompt = self.tool_prompt_prep(
            tools_description, SCENARIOS
        )

        return tool_prompt

        # return DYNAMICAL_TOOL_USAGE.format(tools=tools_description)

    def extract_tool_commands(self, text: str):
        """
        Extract the tool commands from the text

        Example:
        ```json
        {
            "tool": "tool_name",
            "params": {
                "tool1": "inputs",
                "param2": "value2"
            }
        }
        ```

        """
        # Regex to find JSON like strings
        pattern = r"```json(.+?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        json_commands = []
        for match in matches:
            try:
                json_commands = json.loads(match)
                json_commands.append(json_commands)
            except Exception as error:
                print(f"Error parsing JSON command: {error}")

    def parse_and_execute_tools(self, response: str):
        """Parse and execute the tools"""
        json_commands = self.extract_tool_commands(response)
        for command in json_commands:
            tool_name = command.get("tool")
            params = command.get("parmas", {})
            self.execute_tool(tool_name, params)

    def execute_tools(self, tool_name, params):
        """Execute the tool with the provided params"""
        tool = self.tool_find_by_name(tool_name)
        if tool:
            # Execute the tool with the provided parameters
            tool_result = tool.run(**params)
            print(tool_result)

    def truncate_history(self):
        """
        Take the history and truncate it to fit into the model context length
        """
        # truncated_history = self.short_memory[-1][-self.context_length :]
        # self.short_memory[-1] = truncated_history
        # out = limit_tokens_from_string(
        #     "\n".join(truncated_history), self.llm.model_name
        # )
        truncated_history = self.short_memory[-1][
            -self.context_length :
        ]
        text = "\n".join(truncated_history)
        out = limit_tokens_from_string(text, "gpt-4")
        return out

    def add_task_to_memory(self, task: str):
        """Add the task to the memory"""
        self.short_memory.append([f"{self.user_name}: {task}"])

    def add_message_to_memory(self, message: str):
        """Add the message to the memory"""
        self.short_memory[-1].append(message)

    def add_message_to_memory_and_truncate(self, message: str):
        """Add the message to the memory and truncate"""
        self.short_memory[-1].append(message)
        self.truncate_history()

    def print_dashboard(self, task: str):
        """Print dashboard"""
        model_config = self.get_llm_init_params()
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
                    Model Configuration: {model_config}

                ----------------------------------------
                """,
                "green",
            )
        )

    def add_message_to_memory_db(
        self, message: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """Add the message to the memory

        Args:
            message (Dict[str, Any]): _description_
            metadata (Dict[str, Any]): _description_
        """
        if self.memory is not None:
            self.memory.add(message, metadata)

    def query_memorydb(
        self,
        message: Dict[str, Any],
        num_results: int = 100,
    ) -> Dict[str, Any]:
        """Query the memory database

        Args:
            message (Dict[str, Any]): _description_
            num_results (int): _description_

        Returns:
            Dict[str, Any]: _description_
        """
        if self.memory is not None:
            return self.memory.query(message, num_results)
        else:
            return {}

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
            # print(colored("Loading modules...", "yellow"))
            # print(colored("Modules loaded successfully.", "green"))
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

    def agent_system_prompt_2(self):
        """Agent system prompt 2"""
        return agent_system_prompt_2(self.agent_name)

    def run(
        self, task: Optional[str], img: Optional[str] = None, **kwargs
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
            # Activate Autonomous agent message
            self.activate_autonomous_agent()

            response = task  # or combined_prompt
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

                # Check to see if stopping token is in the output to stop the loop
                if self.stopping_token:
                    if self._check_stopping_condition(
                        response
                    ) or parse_done_token(response):
                        break

                # Adjust temperature, comment if no work
                if self.dynamic_temperature_enabled:
                    print(colored("Adjusting temperature...", "blue"))
                    self.dynamic_temperature()

                # Preparing the prompt
                task = self.agent_history_prompt(
                    FLOW_SYSTEM_PROMPT, response
                )

                attempt = 0
                while attempt < self.retry_attempts:
                    try:
                        if img:
                            response = self.llm(
                                task,
                                img,
                                **kwargs,
                            )
                        else:
                            response = self.llm(
                                task,
                                **kwargs,
                            )

                        # If code interpreter is enabled then run the code
                        if self.code_interpreter:
                            self.run_code(response)

                        # If there are any tools then parse and execute them
                        if self.tools:
                            self.parse_and_execute_tools(response)

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
                # Add the response to the history
                history.append(response)

                time.sleep(self.loop_interval)
            # Add the history to the memory
            self.short_memory.append(history)

            # If autosave is enabled then save the state
            if self.autosave:
                print(
                    colored(
                        (
                            "Autosaving agent state to"
                            f" {self.saved_state_path}"
                        ),
                        "green",
                    )
                )
                self.save_state(self.saved_state_path)

            # If return history is enabled then return the response and history
            if self.return_history:
                return response, history

            return response
        except Exception as error:
            print(f"Error running agent: {error}")
            raise

    async def arun(self, task: str, **kwargs):
        """
        Run the autonomous agent loop aschnronously

        Args:
            task (str): The initial task to run

        Agent:
        1. Generate a response
        2. Check stopping condition
        3. If stopping condition is met, stop
        4. If stopping condition is not met, generate a response
        5. Repeat until stopping condition is met or max_loops is reached

        """
        # Activate Autonomous agent message
        self.activate_autonomous_agent()

        response = task
        history = [f"{self.user_name}: {task}"]

        # If dashboard = True then print the dashboard
        if self.dashboard:
            self.print_dashboard(task)

        loop_count = 0
        # for i in range(self.max_loops):
        while self.max_loops == "auto" or loop_count < self.max_loops:
            loop_count += 1
            print(
                colored(
                    f"\nLoop {loop_count} of {self.max_loops}", "blue"
                )
            )
            print("\n")

            if self._check_stopping_condition(
                response
            ) or parse_done_token(response):
                break

            # Adjust temperature, comment if no work
            if self.dynamic_temperature_enabled:
                self.dynamic_temperature()

            # Preparing the prompt
            task = self.agent_history_prompt(
                FLOW_SYSTEM_PROMPT, response
            )

            attempt = 0
            while attempt < self.retry_attempts:
                try:
                    response = self.llm(
                        task**kwargs,
                    )
                    if self.interactive:
                        print(f"AI: {response}")
                        history.append(f"AI: {response}")
                        response = input("You: ")
                        history.append(f"Human: {response}")
                    else:
                        print(f"AI: {response}")
                        history.append(f"AI: {response}")
                        print(response)
                    break
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    attempt += 1
                    time.sleep(self.retry_interval)
            history.append(response)
            time.sleep(self.loop_interval)
        self.memory.append(history)

        if self.autosave:
            print(
                colored(
                    (
                        "Autosaving agent state to"
                        f" {self.saved_state_path}"
                    ),
                    "green",
                )
            )
            self.save_state(self.saved_state_path)

        if self.return_history:
            return response, history

        return response

    def _run(self, **kwargs: Any) -> str:
        """Generate a result using the provided keyword args."""
        task = self.format_prompt(**kwargs)
        response, history = self._generate(task, task)
        logging.info(f"Message history: {history}")
        return response

    def agent_history_prompt(
        self,
        system_prompt: str = FLOW_SYSTEM_PROMPT,
        history=None,
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
            system_prompt = system_prompt or self.system_prompt
            agent_history_prompt = f"""
                SYSTEM_PROMPT: {system_prompt}

                Follow this standard operating procedure (SOP) to complete tasks:
                {self.sop}
                
                -----------------
                ################ CHAT HISTORY ####################
                {history}
            """
            return agent_history_prompt
        else:
            system_prompt = system_prompt or self.system_prompt
            agent_history_prompt = f"""
                SYSTEM_PROMPT: {system_prompt}


                ################ CHAT HISTORY ####################
                {history}
            """
            return agent_history_prompt

    async def run_concurrent(self, tasks: List[str], **kwargs):
        """
        Run a batch of tasks concurrently and handle an infinite level of task inputs.

        Args:
            tasks (List[str]): A list of tasks to run.
        """
        task_coroutines = [
            self.run_async(task, **kwargs) for task in tasks
        ]
        completed_tasks = await asyncio.gather(*task_coroutines)
        return completed_tasks

    def bulk_run(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """Generate responses for multiple input sets."""
        return [self.run(**input_data) for input_data in inputs]

    @staticmethod
    def from_llm_and_template(llm: Any, template: str) -> "Agent":
        """Create AgentStream from LLM and a string template."""
        return Agent(llm=llm, template=template)

    @staticmethod
    def from_llm_and_template_file(
        llm: Any, template_file: str
    ) -> "Agent":
        """Create AgentStream from LLM and a template file."""
        with open(template_file, "r") as f:
            template = f.read()
        return Agent(llm=llm, template=template)

    def save(self, file_path) -> None:
        """Save the agent history to a file.

        Args:
            file_path (_type_): _description_
        """
        with open(file_path, "w") as f:
            json.dump(self.short_memory, f)
        print(f"Saved agent history to {file_path}")

    def load(self, file_path: str):
        """
        Load the agent history from a file.

        Args:
            file_path (str): The path to the file containing the saved agent history.
        """
        with open(file_path, "r") as f:
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
            # Generate the response using lm
            response = self.llm(task, **kwargs)

            # Update the agent's history with the new interaction
            if self.interactive:
                self.short_memory.append(f"AI: {response}")
                self.short_memory.append(f"Human: {task}")
            else:
                self.short_memory.append(f"AI: {response}")

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

        # Remove the last response
        self.short_memory.pop()

        # Get the previous state
        previous_state = self.short_memory[-1][-1]
        return previous_state, f"Restored to {previous_state}"

    # Response Filtering
    def add_response_filter(self, filter_word: str) -> None:
        """
        Add a response filter to filter out certain words from the response

        Example:
        agent.add_response_filter("Trump")
        agent.run("Generate a report on Trump")


        """
        self.reponse_filters.append(filter_word)

    def apply_reponse_filters(self, response: str) -> str:
        """
        Apply the response filters to the response

        """
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
        raw_response = self.run(task)
        return self.apply_response_filters(raw_response)

    def interactive_run(self, max_loops: int = 5) -> None:
        """Interactive run mode"""
        response = input("Start the cnversation")

        for i in range(max_loops):
            ai_response = self.streamed_generation(response)
            print(f"AI: {ai_response}")

            # Get user input
            response = input("You: ")

    def streamed_generation(self, prompt: str) -> str:
        """
        Stream the generation of the response

        Args:
            prompt (str): The prompt to use

        Example:
        # Feature 4: Streamed generation
        response = agent.streamed_generation("Generate a report on finance")
        print(response)

        """
        tokens = list(prompt)
        response = ""
        for token in tokens:
            time.sleep(0.1)
            response += token
            print(token, end="", flush=True)
        print()
        return response

    def get_llm_params(self):
        """
        Extracts and returns the parameters of the llm object for serialization.
        It assumes that the llm object has an __init__ method
        with parameters that can be used to recreate it.
        """
        if not hasattr(self.llm, "__init__"):
            return None

        init_signature = inspect.signature(self.llm.__init__)
        params = init_signature.parameters
        llm_params = {}

        for name, param in params.items():
            if name == "self":
                continue
            if hasattr(self.llm, name):
                value = getattr(self.llm, name)
                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        list,
                        dict,
                        tuple,
                        type(None),
                    ),
                ):
                    llm_params[name] = value
                else:
                    llm_params[name] = str(
                        value
                    )  # For non-serializable objects, save their string representation.

        return llm_params

    def save_state(self, file_path: str) -> None:
        """
        Saves the current state of the agent to a JSON file, including the llm parameters.

        Args:
            file_path (str): The path to the JSON file where the state will be saved.

        Example:
        >>> agent.save_state('saved_flow.json')
        """
        state = {
            "agent_id": str(self.id),
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "system_prompt": self.system_prompt,
            "sop": self.sop,
            "memory": self.short_memory,
            "loop_interval": self.loop_interval,
            "retry_attempts": self.retry_attempts,
            "retry_interval": self.retry_interval,
            "interactive": self.interactive,
            "dashboard": self.dashboard,
            "dynamic_temperature": self.dynamic_temperature_enabled,
            "autosave": self.autosave,
            "saved_state_path": self.saved_state_path,
            "max_loops": self.max_loops,
        }

        with open(file_path, "w") as f:
            json.dump(state, f, indent=4)

        saved = colored(f"Saved agent state to: {file_path}", "green")
        print(saved)

    def load_state(self, file_path: str):
        """
        Loads the state of the agent from a json file and restores the configuration and memory.


        Example:
        >>> agent = Agent(llm=llm_instance, max_loops=5)
        >>> agent.load_state('saved_flow.json')
        >>> agent.run("Continue with the task")

        """
        with open(file_path, "r") as f:
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

    def retry_on_failure(
        self, function, retries: int = 3, retry_delay: int = 1
    ):
        """Retry wrapper for LLM calls."""
        attempt = 0
        while attempt < retries:
            try:
                return function()
            except Exception as error:
                logging.error(f"Error generating response: {error}")
                attempt += 1
                time.sleep(retry_delay)
        raise Exception("All retry attempts failed")

    def generate_reply(self, history: str, **kwargs) -> str:
        """
        Generate a response based on initial or task
        """
        prompt = f"""

        SYSTEM_PROMPT: {self.system_prompt}

        History: {history}

        Your response:
        """
        response = self.llm(prompt, **kwargs)
        return {"role": self.agent_name, "content": response}

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
        self.short_memory = []

    def run_code(self, code: str):
        """
        text -> parse_code by looking for code inside 6 backticks `````-> run_code
        """
        parsed_code = extract_code_in_backticks_in_string(code)
        run_code = self.code_executor.run(parsed_code)
        return run_code

    def pdf_connector(self, pdf: str = None):
        """Transforms the pdf into text

        Args:
            pdf (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        pdf = pdf or self.pdf_path
        text = pdf_to_text(pdf)
        return text

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

    def tools_prompt_prep(
        self, docs: str = None, scenarios: str = None
    ):
        """
        Tools prompt prep

        Args:
            docs (str, optional): _description_. Defaults to None.
            scenarios (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        PROMPT = f"""
        # Task
        You will be provided with a list of APIs. These APIs will have a
        description and a list of parameters and return types for each tool. Your
        task involves creating 3 varied, complex, and detailed user scenarios
        that require at least 5 API calls to complete involving at least 3
        different APIs. One of these APIs will be explicitly provided and the
        other two will be chosen by you.

        For instance, given the APIs: SearchHotels, BookHotel, CancelBooking,
        GetNFLNews. Given that GetNFLNews is explicitly provided, your scenario
        should articulate something akin to:

        "The user wants to see if the Broncos won their last game (GetNFLNews).
        They then want to see if that qualifies them for the playoffs and who
        they will be playing against (GetNFLNews). The Broncos did make it into
        the playoffs, so the user wants watch the game in person. They want to
        look for hotels where the playoffs are occurring (GetNBANews +
        SearchHotels). After looking at the options, the user chooses to book a
        3-day stay at the cheapest 4-star option (BookHotel)."
        13

        This scenario exemplifies a scenario using 5 API calls. The scenario is
        complex, detailed, and concise as desired. The scenario also includes two
        APIs used in tandem, the required API, GetNBANews to search for the
        playoffs location and SearchHotels to find hotels based on the returned
        location. Usage of multiple APIs in tandem is highly desirable and will
        receive a higher score. Ideally each scenario should contain one or more
        instances of multiple APIs being used in tandem.

        Note that this scenario does not use all the APIs given and re-uses the "
        GetNBANews" API. Re-using APIs is allowed, but each scenario should
        involve at least 3 different APIs. Note that API usage is also included
        in the scenario, but exact parameters are not necessary. You must use a
        different combination of APIs for each scenario. All APIs must be used in
        at least one scenario. You can only use the APIs provided in the APIs
        section.
        
        Note that API calls are not explicitly mentioned and their uses are
        included in parentheses. This behaviour should be mimicked in your
        response.
        Deliver your response in this format:
        ‘‘‘
        {scenarios}
        ‘‘‘
        # APIs
        ‘‘‘
        {docs}
        ‘‘‘
        # Response
        ‘‘‘
        """

    def self_healing(self, **kwargs):
        """
        Self healing by debugging errors and refactoring its own code

        Args:
            **kwargs (Any): Any additional keyword arguments
        """
        pass

    def refactor_code(
        self, file: str, changes: List, confirm: bool = False
    ):
        """_summary_

        Args:
            file (str): _description_
            changes (List): _description_
            confirm (bool, optional): _description_. Defaults to False.
        """
        # with open(file) as f:
        #     original_file_lines = f.readlines()

        # # Filter out the changes that are not confirmed
        # operation_changes = [
        #     change for change in changes if "operation" in change
        # ]
        # explanations = [
        #     change["explanation"] for change in changes if "explanation" in change
        # ]

        # Sort the changes in reverse line order
        # explanations.sort(key=lambda x: x["line", reverse=True])
        pass

    def error_prompt_inject(
        self,
        file_path: str,
        args: List,
        error: str,
    ):
        """
        Error prompt injection

        Args:
            file_path (str): _description_
            args (List): _description_
            error (str): _description_

        """
        # with open(file_path, "r") as f:
        #     file_lines = f.readlines()

        # file_with_lines = []
        # for i, line in enumerate(file_lines):
        #     file_with_lines.append(str(i + 1) + "" + line)
        # file_with_lines = "".join(file_with_lines)

        # prompt = f"""
        #     Here is the script that needs fixing:\n\n
        #     {file_with_lines}\n\n
        #     Here are the arguments it was provided:\n\n
        #     {args}\n\n
        #     Here is the error message:\n\n
        #     {error}\n
        #     "Please provide your suggested changes, and remember to stick to the "
        #     exact format as described above.
        #     """
        # print(prompt)
        pass
