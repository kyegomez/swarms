from pydantic import BaseModel
from typing import List, Optional


class TextModality(BaseModel):
    content: str


class ImageModality(BaseModel):
    url: str
    alt_text: Optional[str]


class AudioModality(BaseModel):
    url: str
    transcript: Optional[str]


class VideoModality(BaseModel):
    url: str
    transcript: Optional[str]


class MultimodalData(BaseModel):
    text: Optional[List[TextModality]]
    images: Optional[List[ImageModality]]
    audio: Optional[List[AudioModality]]
    video: Optional[List[VideoModality]]
