import os
import shutil
from pathlib import Path

# from env import DotEnv

from swarms.utils.main import AbstractUploader

class StaticUploader(AbstractUploader):
    def __init__(self, server: str, path: Path, endpoint: str):
        self.server = server
        self.path = path
        self.endpoint = endpoint

    @staticmethod
    def from_settings(path: Path, endpoint: str) -> "StaticUploader":
        return StaticUploader(os.environ["SERVER"], path, endpoint)

    def get_url(self, uploaded_path: str) -> str:
        return f"{self.server}/{uploaded_path}"

    def upload(self, filepath: str):
        relative_path = Path("generated") / filepath.split("/")[-1]
        file_path = self.path / relative_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        shutil.copy(filepath, file_path)
        endpoint_path = self.endpoint / relative_path
        return f"{self.server}/{endpoint_path}"