"""
Flow,
A chain like structure from langchain that provides the autonomy to language models
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

----------------------------------

Example:
from swarms.models import OpenAIChat
from swarms.structs import Flow

# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

# Initialize the flow
flow = Flow(
    llm=llm, max_loops=5,
    #system_prompt=SYSTEM_PROMPT,
    #retry_interval=1,
)

flow.run("Generate a 10,000 word blog")

# Now save the flow
flow.save("path/flow.yaml")

"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from termcolor import colored


# Custome stopping condition
def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "Stop" in response.lower()


class Flow:
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

    def format_prompt(self, template, **kwargs: Any) -> str:
        """Format the template with the provided kwargs using f-string interpolation."""
        return template.format(**kwargs)

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
        for i in range(self.max_loops):
            print(colored(f"\nLoop {i+1} of {self.max_loops}", 'blue'))
            print("\n")
            if self._check_stopping_condition(response):
                break
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
        return response #, history

    def _run(self, **kwargs: Any) -> str:
        """Generate a result using the provided keyword args."""
        task = self.format_prompt(**kwargs)
        response, history = self._generate(task, task)
        logging.info(f"Message history: {history}")
        return response

    def bulk_run(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """Generate responses for multiple input sets."""
        return [self.run(**input_data) for input_data in inputs]

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
        with open(file_path, 'w') as f:
            json.dump(self.memory, f)
        print(f"Saved flow history to {file_path}")

    def load(self, file_path) -> None:
        with open(file_path, 'r') as f:
            self.memory = json.load(f)
        print(f"Loaded flow history from {file_path}")
