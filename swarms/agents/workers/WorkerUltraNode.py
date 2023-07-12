import os
import re
import logging
from pathlib import Path
from typing import Dict, List

from swarms.agents.utils.AgentManager import AgentManager
from swarms.utils.main import BaseHandler, FileHandler, FileType
from swarms.tools.agent_tools import ExitConversation, RequestsGet, CodeEditor, Terminal
from swarms.utils.main import CsvToDataframe

from swarms.tools.main import BaseToolSet
from swarms.utils.main import StaticUploader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR / os.environ["PLAYGROUND_DIR"])

class WorkerUltraNode:
    def __init__(self, objective: str):
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
            from swarms.tools.main import ImageCaptioning
            from swarms.tools.main import ImageEditing, InstructPix2Pix, Text2Image, VisualQuestionAnswering

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


            self.agent_manager = AgentManager.create(toolsets=toolsets)
            self.file_handler = FileHandler(handlers=handlers, path=BASE_DIR)
            self.uploader = StaticUploader.from_settings(
                path=BASE_DIR / "static", endpoint="static"
            )


            self.session = self.agent_manager.create_executor(objective)

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
                
            # The prompt is not needed here either
            return self.execute_task()
        except Exception as e:
            logging.error(f"Error while executing: {str(e)}")
            raise e

# from worker_node import UltraNode

# node = UltraNode('objective')
# result = node.execute()
