import os
from pathlib import Path
from typing import Dict, List

from fastapi.templating import Jinja2Templates

from swarms.agents.utils.agent_creator import AgentManager
from swarms.utils.main import BaseHandler, FileHandler, FileType
from swarms.tools.main import ExitConversation, RequestsGet, CodeEditor, Terminal

from swarms.utils.main import CsvToDataframe

from swarms.tools.main import BaseToolSet

from swarms.utils.main import StaticUploader

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR / os.environ["PLAYGROUND_DIR"])

#
toolsets: List[BaseToolSet] = [
    Terminal(),
    CodeEditor(),
    RequestsGet(),
    ExitConversation(),
]
handlers: Dict[FileType, BaseHandler] = {FileType.DATAFRAME: CsvToDataframe()}

if os.environ["USE_GPU"]:
    import torch

    # from core.handlers.image import ImageCaptioning
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

uploader = StaticUploader.from_settings(path=BASE_DIR / "static", endpoint="static")

reload_dirs = [BASE_DIR / "core", BASE_DIR / "api"]
