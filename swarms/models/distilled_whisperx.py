import asyncio
import os
import time
from functools import wraps
from typing import Union

import torch
from termcolor import colored
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


def async_retry(max_retries=3, exceptions=(Exception,), delay=1):
    """
    A decorator for adding retry logic to async functions.
    :param max_retries: Maximum number of retries before giving up.
    :param exceptions: A tuple of exceptions to catch and retry on.
    :param delay: Delay between retries.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = max_retries
            while retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        raise
                    print(
                        f"Retry after exception: {e}, Attempts"
                        f" remaining: {retries}"
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


class DistilWhisperModel:
    """
    This class encapsulates the Distil-Whisper model for English speech recognition.
    It allows for both synchronous and asynchronous transcription of short and long-form audio.

    Args:
        model_id: The model ID to use. Defaults to "distil-whisper/distil-large-v2".


    Attributes:
        device: The device to use for inference.
        torch_dtype: The torch data type to use for inference.
        model_id: The model ID to use.
        model: The model instance.
        processor: The processor instance.

    Usage:
        model_wrapper = DistilWhisperModel()
        transcription = model_wrapper('path/to/audio.mp3')

        # For async usage
        transcription = asyncio.run(model_wrapper.async_transcribe('path/to/audio.mp3'))
    """

    def __init__(self, model_id="distil-whisper/distil-large-v2"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model_id = model_id
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(self, inputs: Union[str, dict]):
        return self.transcribe(inputs)

    def transcribe(self, inputs: Union[str, dict]):
        """
        Synchronously transcribe the given audio input using the Distil-Whisper model.
        :param inputs: A string representing the file path or a dict with audio data.
        :return: The transcribed text.
        """
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        return pipe(inputs)["text"]

    @async_retry()
    async def async_transcribe(self, inputs: Union[str, dict]):
        """
        Asynchronously transcribe the given audio input using the Distil-Whisper model.
        :param inputs: A string representing the file path or a dict with audio data.
        :return: The transcribed text.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, inputs)

    def real_time_transcribe(self, audio_file_path, chunk_duration=5):
        """
        Simulates real-time transcription of an audio file, processing and printing results
        in chunks with colored output for readability.

        :param audio_file_path: Path to the audio file to be transcribed.
        :param chunk_duration: Duration in seconds of each audio chunk to be processed.
        """
        if not os.path.isfile(audio_file_path):
            print(colored("The audio file was not found.", "red"))
            return

        # Assuming `chunk_duration` is in seconds and `processor` can handle chunk-wise processing
        try:
            with torch.no_grad():
                # Load the whole audio file, but process and transcribe it in chunks
                audio_input = self.processor.audio_file_to_array(
                    audio_file_path
                )
                sample_rate = audio_input.sampling_rate
                len(audio_input.array) / sample_rate
                chunks = [
                    audio_input.array[i : i + sample_rate * chunk_duration]
                    for i in range(
                        0,
                        len(audio_input.array),
                        sample_rate * chunk_duration,
                    )
                ]

                print(
                    colored("Starting real-time transcription...", "green")
                )

                for i, chunk in enumerate(chunks):
                    # Process the current chunk
                    processed_inputs = self.processor(
                        chunk,
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                        padding=True,
                    )
                    processed_inputs = processed_inputs.input_values.to(
                        self.device
                    )

                    # Generate transcription for the chunk
                    logits = self.model.generate(processed_inputs)
                    transcription = self.processor.batch_decode(
                        logits, skip_special_tokens=True
                    )[0]

                    # Print the chunk's transcription
                    print(
                        colored(f"Chunk {i+1}/{len(chunks)}: ", "yellow")
                        + transcription
                    )

                    # Wait for the chunk's duration to simulate real-time processing
                    time.sleep(chunk_duration)

        except Exception as e:
            print(
                colored(
                    f"An error occurred during transcription: {e}",
                    "red",
                )
            )
