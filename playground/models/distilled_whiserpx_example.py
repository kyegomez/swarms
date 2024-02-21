import asyncio

from swarms.models.distilled_whisperx import DistilWhisperModel

model_wrapper = DistilWhisperModel()

# Download mp3 of voice and place the path here
transcription = model_wrapper("path/to/audio.mp3")

# For async usage
transcription = asyncio.run(
    model_wrapper.async_transcribe("path/to/audio.mp3")
)
