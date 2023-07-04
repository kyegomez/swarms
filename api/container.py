
import os
import re
from pathlib import Path
from typing import Dict, List

from fastapi.templating import Jinja2Templates

from swarms.agents.workers.agents import AgentManager
from swarms.tools.main import BaseHandler, FileHandler, FileType
from swarms.tools.main import CsvToDataframe

from swarms.tools.main import BaseToolSet
from swarms.tools.main import ExitConversation, RequestsGet
from swarms.tools.main import CodeEditor

from swarms.tools.main import Terminal
from swarms.tools.main import StaticUploader


BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR / os.getenv["PLAYGROUND_DIR"])


toolsets: List[BaseToolSet] = [
    Terminal(),
    CodeEditor(),
    RequestsGet(),
    ExitConversation(),
]
handlers: Dict[FileType, BaseHandler] = {FileType.DATAFRAME: CsvToDataframe()}

if os.getenv["USE_GPU"]:
    import torch

    from swarms.tools.main import ImageCaptioning
    from swarms.tools.main import (
        ImageEditing,
        InstructPix2Pix,
        Text2Image,
        VisualQuestionAnswering,
    )

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

agent_manager = AgentManager.create(toolsets=toolsets)

file_handler = FileHandler(handlers=handlers, path=BASE_DIR)

templates = Jinja2Templates(directory=BASE_DIR / "api" / "templates")

uploader = StaticUploader.from_settings(
    settings, path=BASE_DIR / "static", endpoint="static"
)

reload_dirs = [BASE_DIR / "swarms", BASE_DIR / "api"]