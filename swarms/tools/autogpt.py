import asyncio
import os
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import torch
from langchain.agents import tool
from langchain.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain.chains.qa_with_sources.loading import (
    BaseCombineDocumentsChain,
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from PIL import Image
from pydantic import Field
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
)

from swarms.utils.logger import logger

ROOT_DIR = "./data/"


@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


@tool
def process_csv(
    llm,
    csv_file_path: str,
    instructions: str,
    output_path: Optional[str] = None,
) -> str:
    """Process a CSV by with pandas in a limited REPL.\
 Only use this after writing data to disk as a csv file.\
 Any figures must be saved to disk to be viewed by the human.\
 Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(
            llm, df, max_iterations=30, verbose=False
        )
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"


async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (
                phrase.strip() for line in lines for phrase in line.split("  ")
            )
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )


class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = (
        "Browse a webpage and retrieve the information relevant to the"
        " question."
    )
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i : i + 4]
            window_result = self.qa_chain(
                {"input_documents": input_docs, "question": question},
                return_only_outputs=True,
            )
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.qa_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
        )

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError


class EdgeGPTTool:
    # Initialize the custom tool
    def __init__(
        self,
        model,
        name="EdgeGPTTool",
        description="Tool that uses EdgeGPTModel to generate responses",
    ):
        super().__init__(name=name, description=description)
        self.model = model

    def _run(self, prompt):
        return self.model.__call__(prompt)


@tool
def VQAinference(self, inputs):
    """
    Answer Question About The Image, VQA Multi-Modal Worker agent
    description="useful when you need an answer for a question based on an image. "
    "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
    "The input to this tool should be a comma separated string of two, representing the image_path and the question",

    """
    device = "cuda:0"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", torch_dtype=torch_dtype
    ).to(device)

    image_path, question = inputs.split(",")
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, question, return_tensors="pt").to(
        device, torch_dtype
    )
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    logger.debug(
        f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input"
        f" Question: {question}, Output Answer: {answer}"
    )

    return answer
