from typing import List, Optional

from pydantic import BaseModel


class TextModality(BaseModel):
    content: str


class ImageModality(BaseModel):
    url: str
    alt_text: Optional[str] = None


class AudioModality(BaseModel):
    url: str
    transcript: Optional[str] = None


class VideoModality(BaseModel):
    url: str
    transcript: Optional[str] = None


class MultimodalData(BaseModel):
    text: Optional[List[TextModality]] = None
    images: Optional[List[ImageModality]] = None
    audio: Optional[List[AudioModality]] = None
    video: Optional[List[VideoModality]] = None
