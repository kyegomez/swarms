"""
TODO:
- Add a retry mechanism
- Add prompt injection letting the agent know it's in a flow, Flow prompt
- Dynamic temperature handling
- Add 

"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Generator
from termcolor import colored
import inspect
import random


# Constants
FLOW_SYSTEM_PROMPT = """
You are a language model operating within a flow class.
Your role is to engage in multi-step conversations with your self or the user, 
generate long-form content like blogs, screenplays, or SOPs, 
and accomplish tasks. You can have internal dialogues with yourself or can interact with the user 
to aid in these complex tasks. Your responses should be coherent, contextually relevant, and tailored to the task at hand.
"""


# Custome stopping condition
def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "Stop" in response.lower()


def parse_done_token(response: str) -> bool:
    """Parse the response to see if the done token is present"""
    return "<DONE>" in response


class Flow:
    """
    Flow is a chain like structure from langchain that provides the autonomy to language models
    to generate sequential responses.

    Features:
    * User defined queries
    * Dynamic keep generating until <DONE> is outputted by the agent
    * Interactive, AI generates, then user input
    * Message history and performance history fed -> into context
    * Ability to save and load flows
    * Ability to provide feedback on responses
    * Ability to provide a stopping condition
    * Ability to provide a retry mechanism
    * Ability to provide a loop interval

    Args:
        llm (Any): The language model to use
        max_loops (int): The maximum number of loops to run
        stopping_condition (Optional[Callable[[str], bool]]): A stopping condition
        loop_interval (int): The interval between loops
        retry_attempts (int): The number of retry attempts
        retry_interval (int): The interval between retry attempts
        interactive (bool): Whether or not to run in interactive mode
        dashboard (bool): Whether or not to print the dashboard
        dynamic_temperature(bool): Dynamical temperature handling
        **kwargs (Any): Any additional keyword arguments

    Example:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import Flow
    >>> llm = OpenAIChat(
    ...     openai_api_key=api_key,
    ...     temperature=0.5,
    ... )
    >>> flow = Flow(
    ...     llm=llm, max_loops=5,
    ...     #system_prompt=SYSTEM_PROMPT,
    ...     #retry_interval=1,
    ... )
    >>> flow.run("Generate a 10,000 word blog")
    >>> flow.save("path/flow.yaml")
    """

    def __init__(
        self,
        llm: Any,
        # template: str,
        max_loops: int = 5,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: int = 1,
        retry_attempts: int = 3,
        retry_interval: int = 1,
        interactive: bool = False,
        dashboard: bool = False,
        dynamic_temperature: bool = False,
        **kwargs: Any,
    ):
        self.llm = llm
        # self.template = template
        self.max_loops = max_loops
        self.stopping_condition = stopping_condition
        self.loop_interval = loop_interval
        self.retry_attempts = retry_attempts
        self.retry_interval = retry_interval
        self.feedback = []
        self.memory = []
        self.task = None
        self.stopping_token = "<DONE>"
        self.interactive = interactive
        self.dashboard = dashboard
        self.dynamic_temperature = dynamic_temperature

    def provide_feedback(self, feedback: str) -> None:
        """Allow users to provide feedback on the responses."""
        self.feedback.append(feedback)
        logging.info(f"Feedback received: {feedback}")

    def _check_stopping_condition(self, response: str) -> bool:
        """Check if the stopping condition is met."""
        if self.stopping_condition:
            return self.stopping_condition(response)
        return False

    def __call__(self, prompt, **kwargs) -> str:
        """Invoke the flow by providing a template and its variables."""
        response = self.llm(prompt, **kwargs)
        return response

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

    def print_dashboard(self, task: str):
        """Print dashboard"""
        model_config = self.get_llm_init_params()

        dashboard = print(
            colored(
                f"""
                Flow Dashboard
                --------------------------------------------

                Flow loop is initializing for {self.max_loops} with the following configuration:

                Model Configuration: {model_config}
                ----------------------------------------

                Flow Configuration:
                    Task: {task}
                    Max Loops: {self.max_loops}
                    Stopping Condition: {self.stopping_condition}
                    Loop Interval: {self.loop_interval}
                    Retry Attempts: {self.retry_attempts}
                    Retry Interval: {self.retry_interval}
                    Interactive: {self.interactive}

                ----------------------------------------
                """,
                "green",
            )
        )

        print(dashboard)

    def run(self, task: str):
        """
        Run the autonomous agent loop

        Args:
            task (str): The initial task to run

        Flow:
        1. Generate a response
        2. Check stopping condition
        3. If stopping condition is met, stop
        4. If stopping condition is not met, generate a response
        5. Repeat until stopping condition is met or max_loops is reached

        """
        response = task
        history = [task]

        # If dashboard = True then print the dashboard
        if self.dashboard:
            self.print_dashboard(task)

        for i in range(self.max_loops):
            print(colored(f"\nLoop {i+1} of {self.max_loops}", "blue"))
            print("\n")
            if self._check_stopping_condition(response):
                break

            # Adjust temperature, comment if no work
            if self.dynamic_temperature:
                self.dynamic_temperature()

            attempt = 0
            while attempt < self.retry_attempts:
                try:
                    response = self.llm(response)
                    print(f"Next query: {response}")
                    break
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    attempt += 1
                    time.sleep(self.retry_interval)
            history.append(response)
            time.sleep(self.loop_interval)
        self.memory.append(history)
        return response  # , history

    def _run(self, **kwargs: Any) -> str:
        """Generate a result using the provided keyword args."""
        task = self.format_prompt(**kwargs)
        response, history = self._generate(task, task)
        logging.info(f"Message history: {history}")
        return response

    def bulk_run(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """Generate responses for multiple input sets."""
        return [self.run(**input_data) for input_data in inputs]

    def run_dynamically(self, task: str, max_loops: Optional[int] = None):
        """
        Run the autonomous agent loop dynamically based on the <DONE>

        # Usage Example

        # Initialize the Flow
        flow = Flow(llm=lambda x: x, max_loops=5)

        # Run dynamically based on <DONE> token and optional max loops
        response = flow.run_dynamically("Generate a report <DONE>", max_loops=3)
        print(response)

        response = flow.run_dynamically("Generate a report <DONE>")
        print(response)

        """
        if "<DONE>" in task:
            self.stopping_condition = parse_done_token
        self.max_loops = max_loops or float("inf")
        response = self.run(task)
        return response

    @staticmethod
    def from_llm_and_template(llm: Any, template: str) -> "Flow":
        """Create FlowStream from LLM and a string template."""
        return Flow(llm=llm, template=template)

    @staticmethod
    def from_llm_and_template_file(llm: Any, template_file: str) -> "Flow":
        """Create FlowStream from LLM and a template file."""
        with open(template_file, "r") as f:
            template = f.read()
        return Flow(llm=llm, template=template)

    def save(self, file_path) -> None:
        with open(file_path, "w") as f:
            json.dump(self.memory, f)
        print(f"Saved flow history to {file_path}")

    def load(self, file_path) -> None:
        with open(file_path, "r") as f:
            self.memory = json.load(f)
        print(f"Loaded flow history from {file_path}")

    def validate_response(self, response: str) -> bool:
        """Validate the response based on certain criteria"""
        if len(response) < 5:
            print("Response is too short")
            return False
        return True

    def run_with_timeout(self, task: str, timeout: int = 60) -> str:
        """Run the loop but stop if it takes longer than the timeout"""
        start_time = time.time()
        response = self.run(task)
        end_time = time.time()
        if end_time - start_time > timeout:
            print("Operaiton timed out")
            return "Timeout"
        return response

    def backup_memory_to_s3(self, bucket_name: str, object_name: str):
        """Backup the memory to S3"""
        import boto3

        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket_name, Key=object_name, Body=json.dumps(self.memory))
        print(f"Backed up memory to S3: {bucket_name}/{object_name}")

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
        response = flow.run("Another task")
        print(f"Response: {response}")
        previous_state, message = flow.undo_last()
        print(message)

        """
        if len(self.memory) < 2:
            return None, None

        # Remove the last response
        self.memory.pop()

        # Get the previous state
        previous_state = self.memory[-1][-1]
        return previous_state, f"Restored to {previous_state}"

    # Response Filtering
    def add_response_filter(self, filter_word: str) -> None:
        """
        Add a response filter to filter out certain words from the response

        Example:
        flow.add_response_filter("Trump")
        flow.run("Generate a report on Trump")


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
        flow.add_response_filter("report")
        response = flow.filtered_run("Generate a report on finance")
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
        response = flow.streamed_generation("Generate a report on finance")
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

    def streamed_token_generation(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate tokens in real-time for a given prompt.

        This method simulates the real-time generation of each token.
        For simplicity, we treat each character of the input as a token
        and yield them with a slight delay. In a real-world scenario,
        this would involve using the LLM's internal methods to generate
        the response token by token.

        Args:
            prompt (str): The input prompt for which the tokens should be generated.

        Yields:
            str: The next token (character) from the generated response.
        """
        tokens = list(prompt)
        for token in tokens:
            time.sleep(0.1)
            yield token
