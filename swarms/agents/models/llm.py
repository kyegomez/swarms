from typing import Optional
import os
import logging

from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.chat_models import ChatOpenAI


# Configure logging-
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLM:
    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 hf_repo_id: Optional[str] = None,
                 hf_api_token: Optional[str] = None,
                 temperature: Optional[float] = 0.5,
                 max_length: Optional[int] = 64):

        # Check if keys are in the environment variables
        openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        hf_api_token = hf_api_token or os.getenv('HUGGINGFACEHUB_API_TOKEN')

        self.openai_api_key = openai_api_key
        self.hf_repo_id = hf_repo_id
        self.hf_api_token = hf_api_token
        self.temperature = temperature
        self.max_length = max_length

        # If the HuggingFace API token is provided, set it in environment variables
        if self.hf_api_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_api_token

        # Initialize the LLM object
        self.initialize_llm()

    def initialize_llm(self):
        model_kwargs = {"temperature": self.temperature, "max_length": self.max_length}
        try:
            if self.hf_repo_id and self.hf_api_token:
                self.llm = HuggingFaceHub(repo_id=self.hf_repo_id, model_kwargs=model_kwargs)
            elif self.openai_api_key:
                self.llm = ChatOpenAI(api_key=self.openai_api_key, model_kwargs=model_kwargs)
            else:
                raise ValueError("Please provide either OpenAI API key or both HuggingFace repository ID and API token.")
        except Exception as e:
            logger.error("Failed to initialize LLM: %s", e)
            raise
        
    def run(self, prompt: str) -> str:
        template = """Question: {question}
        Answer: Let's think step by step."""
        try:
            prompt_template = PromptTemplate(template=template, input_variables=["question"])
            llm_chain = LLMChain(prompt=prompt_template, llm=self.llm)
            return llm_chain.run({"question": prompt})
        except Exception as e:
            logger.error("Failed to generate response: %s", e)
            raise

# # example
# from swarms.utils.llm import LLM
# llm_instance = LLM(openai_api_key="your_openai_key")
# result = llm_instance.run("Who won the FIFA World Cup in 1998?")
# print(result)

# # using HuggingFaceHub
# llm_instance = LLM(hf_repo_id="google/flan-t5-xl", hf_api_token="your_hf_api_token")
# result = llm_instance.run("Who won the FIFA World Cup in 1998?")
# print(result)


# make super easy to chaneg parameters, in class, use cpu and 
#add qlora, 8bit inference
# look into adding deepspeed