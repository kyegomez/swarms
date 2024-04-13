import os
import random
import shutil
import uuid
from abc import ABC, abstractmethod, abstractstaticmethod
from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np
import requests


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except BaseException:
        pass
    return seed


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split("\n")
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(
                paragraphs[0].split(" ")
            )
            paragraphs = paragraphs[1:]
        return "\n" + "\n".join(paragraphs)


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split(".")[0].split("_")
    this_new_uuid = str(uuid.uuid4())[0:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
        recent_prev_file_name = name_split[0]
        new_file_name = f"{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png"
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = f"{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png"
    return os.path.join(head, new_file_name)


def get_new_dataframe_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split(".")[0].split("_")
    this_new_uuid = str(uuid.uuid4())[0:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
        recent_prev_file_name = name_split[0]
        new_file_name = f"{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.csv"
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = f"{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.csv"
    return os.path.join(head, new_file_name)


STATIC_DIR = "static"


class AbstractUploader(ABC):
    @abstractmethod
    def upload(self, filepath: str) -> str:
        pass

    @abstractstaticmethod
    def from_settings() -> "AbstractUploader":
        pass


class FileType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DATAFRAME = "dataframe"
    UNKNOWN = "unknown"

    @staticmethod
    def from_filename(url: str) -> "FileType":
        filename = url.split("?")[0]

        if filename.endswith(".png") or filename.endswith(".jpg"):
            return FileType.IMAGE
        elif filename.endswith(".mp3") or filename.endswith(".wav"):
            return FileType.AUDIO
        elif filename.endswith(".mp4") or filename.endswith(".avi"):
            return FileType.VIDEO
        elif filename.endswith(".csv"):
            return FileType.DATAFRAME
        else:
            return FileType.UNKNOWN

    @staticmethod
    def from_url(url: str) -> "FileType":
        return FileType.from_filename(url.split("?")[0])

    def to_extension(self) -> str:
        if self == FileType.IMAGE:
            return ".png"
        elif self == FileType.AUDIO:
            return ".mp3"
        elif self == FileType.VIDEO:
            return ".mp4"
        elif self == FileType.DATAFRAME:
            return ".csv"
        else:
            return ".unknown"


class BaseHandler:
    def handle(self, filename: str) -> str:
        raise NotImplementedError


class FileHandler:
    def __init__(
        self, handlers: Dict[FileType, BaseHandler], path: Path
    ):
        self.handlers = handlers
        self.path = path

    def register(
        self, filetype: FileType, handler: BaseHandler
    ) -> "FileHandler":
        self.handlers[filetype] = handler
        return self

    def download(self, url: str) -> str:
        filetype = FileType.from_url(url)
        data = requests.get(url).content
        local_filename = os.path.join(
            "file", str(uuid.uuid4())[0:8] + filetype.to_extension()
        )
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        with open(local_filename, "wb") as f:
            size = f.write(data)
        print(f"Inputs: {url} ({size // 1000}MB)  => {local_filename}")
        return local_filename

    def handle(self, url: str) -> str:
        try:
            if url.startswith(
                os.environ.get("SERVER", "http://localhost:8000")
            ):
                local_filepath = url[
                    len(
                        os.environ.get(
                            "SERVER", "http://localhost:8000"
                        )
                    )
                    + 1:
                ]
                local_filename = (
                    Path("file") / local_filepath.split("/")[-1]
                )
                src = self.path / local_filepath
                dst = (
                    self.path
                    / os.environ.get("PLAYGROUND_DIR", "./playground")
                    / local_filename
                )
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
            else:
                local_filename = self.download(url)
            handler = self.handlers.get(FileType.from_url(url))
            if handler is None:
                if FileType.from_url(url) == FileType.IMAGE:
                    raise Exception(
                        f"No handler for {FileType.from_url(url)}."
                        " Please set USE_GPU to True in"
                        " env/settings.py"
                    )
                else:
                    raise Exception(
                        f"No handler for {FileType.from_url(url)}"
                    )
            return handler.handle(local_filename)
        except Exception as e:
            raise e


# =>  base end

# ===========================>
