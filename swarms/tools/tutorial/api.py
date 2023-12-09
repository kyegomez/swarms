import os
import random
import requests
import hashlib
from ..tool import Tool

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI


def build_tool(config) -> Tool:
    tool = Tool(
        tool_name="Tutorial",
        description="Provide tutorial for foundation model based on a given objective.",
        name_for_model="Tutorial",
        description_for_model="Plugin for providing tutorial for a given objective.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="xin.cong@outlook.com",
        legal_info_url="hello@legal.com",
    )
    prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    )

    key = os.environ.get("OPENAI_API_KEY")
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=key)

    chain = LLMChain(llm=llm, prompt=prompt)

    @tool.get("/tutorial")
    def tutorial(text: str) -> str:
        """
        tutorial(text: str) -> str: Providing a TODO list as a toturial for the foundation model based on the given objective.
        """
        result = chain.run(text)
        return result

    return tool
