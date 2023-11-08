from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


class MetaPrompterAgent:
    """
    Meta Prompting Agent
    The Meta Prompting Agent has 1 purpose: to create better prompts for an agent.

    The meta prompting agent would be used in this flow:
    user task -> MetaPrompterAgent -> Agent

    Args:
        llm (BaseLanguageModel): Language Model
        max_iters (int, optional): Maximum number of iterations. Defaults to 3.
        max_meta_iters (int, optional): Maximum number of meta iterations. Defaults to 5.
        failed_phrase (str, optional): Phrase to indicate failure. Defaults to "task failed".
        success_phrase (str, optional): Phrase to indicate success. Defaults to "task succeeded".
        instructions (str, optional): Instructions to be used in the meta prompt. Defaults to "None".
        template (str, optional): Template to be used in the meta prompt. Defaults to None.
        memory (ConversationBufferWindowMemory, optional): Memory to be used in the meta prompt. Defaults to None.
        meta_template (str, optional): Template to be used in the meta prompt. Defaults to None.
        human_input (bool, optional): Whether to use human input. Defaults to False.

    Returns:
        str: Response from the agent

    Usage:
    --------------
    from swarms.workers import Worker
    from swarms.agents.meta_prompter import MetaPrompterAgent
    from langchain.llms import OpenAI

    #init llm
    llm = OpenAI()

    #init the meta prompter agent that optimized prompts
    meta_optimizer = MetaPrompterAgent(llm=llm)

    #init the worker agent
    worker = Worker(llm)

    #broad task to complete
    task = "Create a feedforward in pytorch"

    #optimize the prompt
    optimized_prompt = meta_optimizer.run(task)

    #run the optimized prompt with detailed instructions
    result = worker.run(optimized_prompt)

    print(result)
    """

    def __init__(
        self,
        llm,
        max_iters: int = 3,
        max_meta_iters: int = 5,
        failed_phrase: str = "task failed",
        success_phrase: str = "task succeeded",
        instructions: str = "None",
        template: str = None,
        memory=None,
        meta_template: str = None,
        human_input: bool = False,
    ):
        self.llm = llm
        self.max_iters = max_iters
        self.max_meta_iters = max_meta_iters
        self.failed_phrase = failed_phrase
        self.success_phrase = success_phrase
        self.instructions = instructions
        self.template = template
        self.memory = memory
        self.meta_template = meta_template
        self.human_input = human_input

        if memory is None:
            memory = ConversationBufferWindowMemory()
            memory.ai_prefix = "Assistant:"

        template = f"""
        Instructions: {self.instructions}
        {{{memory.memory_key}}}
        Human: {{human_input}}
        Assistant:
        """

        prompt = PromptTemplate(input_variables=["history", "human_input"],
                                template=template)

        self.chain = LLMChain(
            llm=self.llm(),
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(),
        )

    def get_chat_history(self, chain_memory):
        """Get Chat History from the memory"""
        memory_key = chain_memory.memory_key
        chat_history = chain_memory.load_memory_variables(
            memory_key)[memory_key]
        return chat_history

    def get_new_instructions(self, meta_output):
        """Get New Instructions from the meta_output"""
        delimiter = "Instructions: "
        new_instructions = meta_output[meta_output.find(delimiter) +
                                       len(delimiter):]
        return new_instructions

    def run(self, task: str):
        """
        Run the MetaPrompterAgent

        Args:
            task (str): The task to be completed

        Returns:
            str: The response from the agent
        """
        key_phrases = [self.success_phrase, self.failed_phrase]

        for i in range(self.max_meta_iters):
            print(f"[Epsisode: {i+1}/{self.max_meta_iters}]")

            chain = self.chain(memory=None)

            output = chain.predict(human_input=task)

            for j in range(self.max_iters):
                print(f"(Step {j+1}/{self.max_iters})")
                print(f"Assistant: {output}")
                print("Human: ")

                if self.human_input:
                    human_input = input()

                if any(phrase in human_input.lower() for phrase in key_phrases):
                    break

                output = chain.predict(human_input.lower)

            if self.success_phrase in human_input.lower():
                print("You succeed! Thanks for using!")
                return

            meta_chain = self.initialize_meta_chain()
            meta_output = meta_chain.predict(
                chat_history=self.get_chat_history(chain.memory))
            print(f"Feedback: {meta_output}")

            self.instructions = self.get_new_instructions(meta_output)
            print(f"New Instruction: {self.instructions}")
            print("\n" + "#" * 80 + "\n")
