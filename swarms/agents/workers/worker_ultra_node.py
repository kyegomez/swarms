import os
import re
import logging
from pathlib import Path
from typing import Dict, List

from swarms.agents.utils.agent_creator import AgentCreator
from swarms.utils.main import BaseHandler, FileHandler, FileType
from swarms.agents.tools.main import ExitConversation, RequestsGet, CodeEditor, Terminal
from swarms.utils.main import CsvToDataframe
from swarms.agents.tools.main import BaseToolSet
from swarms.utils.main import StaticUploader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent

# Check if "PLAYGROUND_DIR" environment variable exists, if not, set a default value
playground = os.environ.get("PLAYGROUND_DIR", './playground')

# Ensure the path exists before changing the directory
os.makedirs(BASE_DIR / playground, exist_ok=True)

try:
    os.chdir(BASE_DIR / playground)
except Exception as e:
    logging.error(f"Failed to change directory: {e}")

class WorkerUltraNode:
    def __init__(self, objective: str, openai_api_key: str):
        self.openai_api_key = openai_api_key 

        if not isinstance(objective, str):
            raise TypeError("Objective must be a string")
        if not objective:
            raise ValueError("Objective cannot be empty")
        
        toolsets: List[BaseToolSet] = [
            Terminal(),
            CodeEditor(),
            RequestsGet(),
            ExitConversation(),
        ]
        handlers: Dict[FileType, BaseHandler] = {FileType.DATAFRAME: CsvToDataframe()}

        if os.environ.get("USE_GPU", False):
            import torch
            from swarms.agents.tools.main import ImageCaptioning
            from swarms.agents.tools.main import ImageEditing, InstructPix2Pix, Text2Image, VisualQuestionAnswering

            if torch.cuda.is_available():
                toolsets.extend(
                    [
                        Text2Image("cuda"),
                        ImageEditing("cuda"),
                        InstructPix2Pix("cuda"),
                        VisualQuestionAnswering("cuda"),
                    ]
                )
                handlers[FileType.IMAGE] = ImageCaptioning("cuda")

        try:
            self.agent_manager = AgentCreator.create(toolsets=toolsets)
            self.file_handler = FileHandler(handlers=handlers, path=BASE_DIR)
            self.uploader = StaticUploader.from_settings(
                path=BASE_DIR / "static", endpoint="static"
            )
            self.session = self.agent_manager.create_executor(objective, self.openai_api_key)

        except Exception as e:
            logging.error(f"Error while initializing WorkerUltraNode: {str(e)}")
            raise e

    def execute_task(self):
        # Now the prompt is not needed as an argument
        promptedQuery = self.file_handler.handle(self.objective)

        try:
            res = self.session({"input": promptedQuery})
        except Exception as e:
            logging.error(f"Error while executing task: {str(e)}")
            return {"answer": str(e), "files": []}

        files = re.findall(r"\[file://\S*\]", res["output"])
        files = [file[1:-1].split("file://")[1] for file in files]

        return {
            "answer": res["output"],
            "files": [self.uploader.upload(file) for file in files],
        }

    def execute(self):
        try:
            return self.execute_task()
        except Exception as e:
            logging.error(f"Error while executing: {str(e)}")
            raise e

class WorkerUltra:
    def __init__(self, objective, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as argument or as an environment variable named 'OPENAI_API_KEY'.")
        self.worker_node = WorkerUltraNode(objective, self.api_key)

    def execute(self):
        try:
            return self.worker_node.execute_task()
        except Exception as e:
            logging.error(f"Error while executing: {str(e)}")
            raise e