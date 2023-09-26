from concurrent.futures import ThreadPoolExecutor
from termcolor import colored
from tabulate import tabulate
import anthropic
from langchain.llms import Anthropic

class GodMode:
    def __init__(self, llms):
        self.llms = llms

    def run_all(self, task):
        with ThreadPoolExecutor() as executor:
            responses = executor.map(lambda llm: llm(task), self.llms)
        return list(responses)

    def print_responses(self, task):
        responses = self.run_all(task)
        table = []
        for i, response in enumerate(responses):
            table.append([f"LLM {i+1}", response])
        print(colored(tabulate(table, headers=["LLM", "Response"], tablefmt="pretty"), "cyan"))

# Usage
llms = [Anthropic(model="<model_name>", anthropic_api_key="my-api-key") for _ in range(5)]

god_mode = GodMode(llms)
task = f"{anthropic.HUMAN_PROMPT} What are the biggest risks facing humanity?{anthropic.AI_PROMPT}"
god_mode.print_responses(task)