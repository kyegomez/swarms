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
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

import yaml


# Custome stopping condition
def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "Stop" in response.lower()


class Flow:
    def __init__(
        self,
        llm: Any,
        # template: str,
        max_loops: int = 1,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: int = 1,
        retry_attempts: int = 3,
        retry_interval: int = 1,
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

    def run(self, task: str):  # formatted_prompts: str) -> str:
        """
        Generate a result using the lm with optional query loops and stopping conditions.
        """
        response = task
        history = [task]
        for _ in range(self.max_loops):
            if self._check_stopping_condition(response):
                break
            attempt = 0
            while attempt < self.retry_attempts:
                try:
                    response = self.llm(response)
                    break
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    attempt += 1
                    time.sleep(self.retry_interval)
            logging.info(f"Generated response: {response}")
            history.append(response)
            time.sleep(self.loop_interval)
        return response, history

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
        """Save the flow.

        Expects `Flow._flow_type` property to be implemented and for memory to be
            null.

        Args:
            file_path: Path to file to save the flow to.

        Example:
            .. code-block:: python

                flow.save(file_path="path/flow.yaml")


        TODO: Save memory list and not dict.
        """
        if self.memory is not None:
            raise ValueError("Saving of memory is not yet supported.")

        # Fetch dictionary to save
        flow_dict = self.dict()
        if "_type" not in flow_dict:
            raise NotImplementedError(f"Flow {self} does not support saving.")

        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(flow_dict, f, indent=4)
                print(f"Saved Flow to JSON file: {file_path}")
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(flow_dict, f, default_flow_style=False)
                print(f"Saved flow history to {file_path} as YAML")
        else:
            raise ValueError(f"{save_path} must be json or yaml")
