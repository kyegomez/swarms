import os
from pathlib import Path
from typing import Dict, List

from swarms.agents.utils.manager import AgentManager
from swarms.utils.utils import BaseHandler, FileHandler, FileType
from swarms.tools.main import CsvToDataframe, ExitConversation, RequestsGet, CodeEditor, Terminal
from swarms.tools.main import BaseToolSet
from swarms.utils.utils import StaticUploader

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR / os.environ["PLAYGROUND_DIR"])

class UltraNode:
    def __init__(self, objective: str):
        toolsets: List[BaseToolSet] = [
            Terminal(),
            CodeEditor(),
            RequestsGet(),
            ExitConversation(),
        ]
        handlers: Dict[FileType, BaseHandler] = {FileType.DATAFRAME: CsvToDataframe()}

        if os.environ["USE_GPU"]:
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

        self.agent_manager = AgentManager.create(toolsets=toolsets)
        self.file_handler = FileHandler(handlers=handlers, path=BASE_DIR)
        self.uploader = StaticUploader.from_settings(
            path=BASE_DIR / "static", endpoint="static"
        )


        self.session = self.agent_manager.create_executor(objective)

    def execute_task(self):
        # Now the prompt is not needed as an argument
        promptedQuery = self.file_handler.handle(self.objective)

        try:
            res = self.session({"input": promptedQuery})
        except Exception as e:
            return {"answer": str(e), "files": []}

        files = re.findall(r"\[file://\S*\]", res["output"])
        files = [file[1:-1].split("file://")[1] for file in files]

        return {
            "answer": res["output"],
            "files": [self.uploader.upload(file) for file in files],
        }
    

    def execute(self):
        # The prompt is not needed here either
        return self.execute_task()

from worker_node import UltraNode

node = UltraNode('objective')
result = node.execute()
