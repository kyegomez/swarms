import os
import random
import uuid

import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
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
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(" "))
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
        new_file_name = "{}_{}_{}_{}.png".format(
            this_new_uuid, func_name, recent_prev_file_name, most_org_file_name
        )
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = "{}_{}_{}_{}.png".format(
            this_new_uuid, func_name, recent_prev_file_name, most_org_file_name
        )
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
        new_file_name = "{}_{}_{}_{}.csv".format(
            this_new_uuid, func_name, recent_prev_file_name, most_org_file_name
        )
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = "{}_{}_{}_{}.csv".format(
            this_new_uuid, func_name, recent_prev_file_name, most_org_file_name
        )
    return os.path.join(head, new_file_name)
#########=======================> utils end








#########=======================> ANSI BEGINNING


class Code:
    def __init__(self, value: int):
        self.value = value

    def __str__(self):
        return "%d" % self.value


class Color(Code):
    def bg(self) -> "Color":
        self.value += 10
        return self

    def bright(self) -> "Color":
        self.value += 60
        return self

    @staticmethod
    def black() -> "Color":
        return Color(30)

    @staticmethod
    def red() -> "Color":
        return Color(31)

    @staticmethod
    def green() -> "Color":
        return Color(32)

    @staticmethod
    def yellow() -> "Color":
        return Color(33)

    @staticmethod
    def blue() -> "Color":
        return Color(34)

    @staticmethod
    def magenta() -> "Color":
        return Color(35)

    @staticmethod
    def cyan() -> "Color":
        return Color(36)

    @staticmethod
    def white() -> "Color":
        return Color(37)

    @staticmethod
    def default() -> "Color":
        return Color(39)


class Style(Code):
    @staticmethod
    def reset() -> "Style":
        return Style(0)

    @staticmethod
    def bold() -> "Style":
        return Style(1)

    @staticmethod
    def dim() -> "Style":
        return Style(2)

    @staticmethod
    def italic() -> "Style":
        return Style(3)

    @staticmethod
    def underline() -> "Style":
        return Style(4)

    @staticmethod
    def blink() -> "Style":
        return Style(5)

    @staticmethod
    def reverse() -> "Style":
        return Style(7)

    @staticmethod
    def conceal() -> "Style":
        return Style(8)


class ANSI:
    ESCAPE = "\x1b["
    CLOSE = "m"

    def __init__(self, text: str):
        self.text = text
        self.args = []

    def join(self) -> str:
        return ANSI.ESCAPE + ";".join([str(a) for a in self.args]) + ANSI.CLOSE

    def wrap(self, text: str) -> str:
        return self.join() + text + ANSI(Style.reset()).join()

    def to(self, *args: str):
        self.args = list(args)
        return self.wrap(self.text)


def dim_multiline(message: str) -> str:
    lines = message.split("\n")
    if len(lines) <= 1:
        return lines[0]
    return lines[0] + ANSI("\n... ".join([""] + lines[1:])).to(Color.black().bright())

#+=============================> ANSI Ending


#================================> upload base

from abc import ABC, abstractmethod, abstractstaticmethod

from env import DotEnv

STATIC_DIR = "static"


class AbstractUploader(ABC):
    @abstractmethod
    def upload(self, filepath: str) -> str:
        pass

    @abstractstaticmethod
    def from_settings(settings: DotEnv) -> "AbstractUploader":
        pass

#================================> upload end


#========================= upload s3
import os

import boto3

from env import DotEnv


class S3Uploader(AbstractUploader):
    def __init__(self, accessKey: str, secretKey: str, region: str, bucket: str):
        self.accessKey = accessKey
        self.secretKey = secretKey
        self.region = region
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            aws_access_key_id=self.accessKey,
            aws_secret_access_key=self.secretKey,
        )

    @staticmethod
    def from_settings(settings: DotEnv) -> "S3Uploader":
        return S3Uploader(
            settings["AWS_ACCESS_KEY_ID"],
            settings["AWS_SECRET_ACCESS_KEY"],
            settings["AWS_REGION"],
            settings["AWS_S3_BUCKET"],
        )

    def get_url(self, object_name: str) -> str:
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{object_name}"

    def upload(self, filepath: str) -> str:
        object_name = os.path.basename(filepath)
        self.client.upload_file(filepath, self.bucket, object_name)
        return self.get_url(object_name)

#========================= upload s3

#========================> upload/static
import os
import shutil
from pathlib import Path

from env import DotEnv


class StaticUploader(AbstractUploader):
    def __init__(self, server: str, path: Path, endpoint: str):
        self.server = server
        self.path = path
        self.endpoint = endpoint

    @staticmethod
    def from_settings(settings: DotEnv, path: Path, endpoint: str) -> "StaticUploader":
        return StaticUploader(settings["SERVER"], path, endpoint)

    def get_url(self, uploaded_path: str) -> str:
        return f"{self.server}/{uploaded_path}"

    def upload(self, filepath: str):
        relative_path = Path("generated") / filepath.split("/")[-1]
        file_path = self.path / relative_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        shutil.copy(filepath, file_path)
        endpoint_path = self.endpoint / relative_path
        return f"{self.server}/{endpoint_path}"