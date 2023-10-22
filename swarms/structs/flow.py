import time
from typing import Any, Dict, List, Optional, Union, Callable
from swarms.models import OpenAIChat
from typing import Any, Dict, List, Optional, Callable
import logging
import time


# Custome stopping condition
def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "Stop" in response.lower()


# class Flow:
#     def __init__(
#         self,
#         llm: Any,
#         template: str,
#         max_loops: int = 1,
#         stopping_condition: Optional[Callable[[str], bool]] = None,
#         **kwargs: Any
#     ):
#         self.llm = llm
#         self.template = template
#         self.max_loops = max_loops
#         self.stopping_condition = stopping_condition
#         self.feedback = []
#         self.history = []

#     def __call__(
#         self,
#         prompt,
#         **kwargs
#     ) -> str:
#         """Invoke the flow by providing a template and it's variables"""
#         response = self.llm(prompt, **kwargs)
#         return response

#     def _check_stopping_condition(self, response: str) -> bool:
#         """Check if the stopping condition is met"""
#         if self.stopping_condition:
#             return self.stopping_condition(response)
#         return False


#     def provide_feedback(self, feedback: str) -> None:
#         """Allow users to to provide feedback on the responses"""
#         feedback = self.feedback.append(feedback)
#         return feedback

#     def format_prompt(self, **kwargs: Any) -> str:
#         """Format the template with the provided kwargs using f string interpolation"""
#         return self.template.format(**kwargs)

#     def _generate(self, formatted_prompts: str) -> str:
#         """
#         Generate a result using the lm

#         """
#         return self.llm(formatted_prompts)

#     def run(self, **kwargs: Any) -> str:
#         """Generate a result using the provided keyword args"""
#         prompt = self.format_prompt(**kwargs)
#         response = self._generate(prompt)
#         return response

#     def bulk_run(
#         self,
#         inputs: List[Dict[str, Any]]
#     ) -> List[str]:
#         """Generate responses for multiple input sets"""
#         return [self.run(**input_data) for input_data in inputs]

#     @staticmethod
#     def from_llm_and_template(llm: Any, template: str) -> "Flow":
#         """Create FlowStream from LLM and a string template"""
#         return Flow(llm=llm, template=template)

#     @staticmethod
#     def from_llm_and_template_file(llm: Any, template_file: str) -> "Flow":
#         """Create FlowStream from LLM and a template file"""
#         with open(template_file, "r") as f:
#             template = f.read()

#         return Flow(llm=llm, template=template)



class Flow:
    def __init__(
        self,
        llm: Any,
        template: str,
        max_loops: int = 1,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: int = 1,
        retry_attempts: int = 3,
        retry_interval: int = 1,
        **kwargs: Any,
    ):
        self.llm = llm
        self.template = template
        self.max_loops = max_loops
        self.stopping_condition = stopping_condition
        self.loop_interval = loop_interval
        self.retry_attempts = retry_attempts
        self.retry_interval = retry_interval
        self.feedback = []

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

    def format_prompt(self, **kwargs: Any) -> str:
        """Format the template with the provided kwargs using f-string interpolation."""
        return self.template.format(**kwargs)

    def _generate(self, task: str, formatted_prompts: str) -> str:
        """
        Generate a result using the lm with optional query loops and stopping conditions.
        """
        response = formatted_prompts
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

    def run(self, **kwargs: Any) -> str:
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


# # Configure logging
# logging.basicConfig(level=logging.INFO)

# llm = OpenAIChat(
#     api_key="YOUR_API_KEY",
#     max_tokens=1000,
#     temperature=0.9,
# )


# def main():
#     # Initialize the Flow class with parameters
#     flow = Flow(
#         llm=llm,
#         template="Translate this to backwards: {sentence}",
#         max_loops=3,
#         stopping_condition=stop_when_repeats,
#         loop_interval=2,  # Wait 2 seconds between loops
#         retry_attempts=2,
#         retry_interval=1,  # Wait 1 second between retries
#     )

#     # Predict using the Flow
#     response = flow.run(sentence="Hello, World!")
#     print("Response:", response)
#     time.sleep(1)  # Pause for demonstration purposes

#     # Provide feedback on the result
#     flow.provide_feedback("The translation was interesting!")
#     time.sleep(1)  # Pause for demonstration purposes

#     # Bulk run
#     inputs = [
#         {"sentence": "This is a test."},
#         {"sentence": "OpenAI is great."},
#         {"sentence": "GPT models are powerful."},
#         {"sentence": "stop and check if our stopping condition works."},
#     ]

#     responses = flow.bulk_run(inputs=inputs)
#     for idx, res in enumerate(responses):
#         print(f"Input: {inputs[idx]['sentence']}, Response: {res}")
#         time.sleep(1)  # Pause for demonstration purposes


# if __name__ == "__main__":
#     main()
