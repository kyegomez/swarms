import os
import re
from pathlib import Path
from typing import Dict, List

from fastapi.templating import Jinja2Templates

from swarms import Swarms
from swarms.utils.utils import BaseHandler, FileHandler, FileType, StaticUploader, CsvToDataframe

from swarms.tools.main import BaseToolSet, ExitConversation, RequestsGet, CodeEditor, Terminal


BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE_DIR / os.getenv("PLAYGROUND_DIR"))

api_key = os.getenv("OPENAI_API_KEY")

toolsets: List[BaseToolSet] = [
    Terminal(),
    CodeEditor(),
    RequestsGet(),
    ExitConversation(),
]
handlers: Dict[FileType, BaseHandler] = {FileType.DATAFRAME: CsvToDataframe()}

if os.getenv("USE_GPU") == "True":
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

swarms = Swarms(api_key)

file_handler = FileHandler(handlers=handlers, path=BASE_DIR)

templates = Jinja2Templates(directory=BASE_DIR / "api" / "templates")

uploader = StaticUploader(
    static_dir=BASE_DIR / "static",
    endpoint="static",
    public_url=os.getenv("PUBLIC_URL")
)

reload_dirs = [BASE_DIR / "swarms", BASE_DIR / "api"]
