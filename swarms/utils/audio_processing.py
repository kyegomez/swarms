import base64
from typing import Union, Dict, Any, Tuple
import requests
from pathlib import Path
import wave
import numpy as np


def encode_audio_to_base64(audio_path: Union[str, Path]) -> str:
    """
    Encode a WAV file to base64 string.

    Args:
        audio_path (Union[str, Path]): Path to the WAV file

    Returns:
        str: Base64 encoded string of the audio file

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        ValueError: If the file is not a valid WAV file
    """
    try:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if not audio_path.suffix.lower() == ".wav":
            raise ValueError("File must be a WAV file")

        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        raise Exception(f"Error encoding audio file: {str(e)}")


def decode_base64_to_audio(
    base64_string: str, output_path: Union[str, Path]
) -> None:
    """
    Decode a base64 string to a WAV file.

    Args:
        base64_string (str): Base64 encoded audio data
        output_path (Union[str, Path]): Path where the WAV file should be saved

    Raises:
        ValueError: If the base64 string is invalid
        IOError: If there's an error writing the file
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_data = base64.b64decode(base64_string)
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)
    except Exception as e:
        raise Exception(f"Error decoding audio data: {str(e)}")


def download_audio_from_url(
    url: str, output_path: Union[str, Path]
) -> None:
    """
    Download an audio file from a URL and save it locally.

    Args:
        url (str): URL of the audio file
        output_path (Union[str, Path]): Path where the audio file should be saved

    Raises:
        requests.RequestException: If there's an error downloading the file
        IOError: If there's an error saving the file
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url)
        response.raise_for_status()

        with open(output_path, "wb") as audio_file:
            audio_file.write(response.content)
    except Exception as e:
        raise Exception(f"Error downloading audio file: {str(e)}")


def process_audio_with_model(
    audio_path: Union[str, Path],
    model: str,
    prompt: str,
    voice: str = "alloy",
    format: str = "wav",
) -> Dict[str, Any]:
    """
    Process an audio file with a model that supports audio input/output.

    Args:
        audio_path (Union[str, Path]): Path to the input WAV file
        model (str): Model name to use for processing
        prompt (str): Text prompt to accompany the audio
        voice (str, optional): Voice to use for audio output. Defaults to "alloy"
        format (str, optional): Audio format. Defaults to "wav"

    Returns:
        Dict[str, Any]: Model response containing both text and audio if applicable

    Raises:
        ImportError: If litellm is not installed
        ValueError: If the model doesn't support audio processing
    """
    try:
        from litellm import (
            completion,
            supports_audio_input,
            supports_audio_output,
        )

        if not supports_audio_input(model):
            raise ValueError(
                f"Model {model} does not support audio input"
            )

        # Encode the audio file
        encoded_audio = encode_audio_to_base64(audio_path)

        # Prepare the messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio,
                            "format": format,
                        },
                    },
                ],
            }
        ]

        # Make the API call
        response = completion(
            model=model,
            modalities=["text", "audio"],
            audio={"voice": voice, "format": format},
            messages=messages,
        )

        return response
    except ImportError:
        raise ImportError(
            "Please install litellm: pip install litellm"
        )
    except Exception as e:
        raise Exception(
            f"Error processing audio with model: {str(e)}"
        )


def read_wav_file(
    file_path: Union[str, Path],
) -> Tuple[np.ndarray, int]:
    """
    Read a WAV file and return its audio data and sample rate.

    Args:
        file_path (Union[str, Path]): Path to the WAV file

    Returns:
        Tuple[np.ndarray, int]: Audio data as numpy array and sample rate

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid WAV file
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {file_path}"
            )

        with wave.open(str(file_path), "rb") as wav_file:
            # Get audio parameters
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            # Read audio data
            frames = wav_file.readframes(n_frames)

            # Convert to numpy array
            dtype = np.int16 if sample_width == 2 else np.int8
            audio_data = np.frombuffer(frames, dtype=dtype)

            # Reshape if stereo
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2)

            return audio_data, frame_rate

    except Exception as e:
        raise Exception(f"Error reading WAV file: {str(e)}")


def write_wav_file(
    audio_data: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int,
    sample_width: int = 2,
) -> None:
    """
    Write audio data to a WAV file.

    Args:
        audio_data (np.ndarray): Audio data as numpy array
        file_path (Union[str, Path]): Path where to save the WAV file
        sample_rate (int): Sample rate of the audio
        sample_width (int, optional): Sample width in bytes. Defaults to 2 (16-bit)

    Raises:
        ValueError: If the audio data is invalid
        IOError: If there's an error writing the file
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure audio data is in the correct format
        if audio_data.dtype != np.int16 and sample_width == 2:
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int8 and sample_width == 1:
            audio_data = (audio_data * 127).astype(np.int8)

        # Determine number of channels
        n_channels = (
            2
            if len(audio_data.shape) > 1 and audio_data.shape[1] == 2
            else 1
        )

        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

    except Exception as e:
        raise Exception(f"Error writing WAV file: {str(e)}")


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalize audio data to have maximum amplitude of 1.0.

    Args:
        audio_data (np.ndarray): Input audio data

    Returns:
        np.ndarray: Normalized audio data
    """
    return audio_data / np.max(np.abs(audio_data))


def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.

    Args:
        audio_data (np.ndarray): Input audio data (stereo)

    Returns:
        np.ndarray: Mono audio data
    """
    if len(audio_data.shape) == 1:
        return audio_data
    return np.mean(audio_data, axis=1)


def encode_wav_to_base64(
    audio_data: np.ndarray, sample_rate: int
) -> str:
    """
    Convert audio data to base64 encoded WAV string.

    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate of the audio

    Returns:
        str: Base64 encoded WAV data
    """
    # Create a temporary WAV file in memory
    with wave.open("temp.wav", "wb") as wav_file:
        wav_file.setnchannels(1 if len(audio_data.shape) == 1 else 2)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    # Read the file and encode to base64
    with open("temp.wav", "rb") as f:
        wav_bytes = f.read()

    # Clean up temporary file
    Path("temp.wav").unlink()

    return base64.b64encode(wav_bytes).decode("utf-8")


def decode_base64_to_wav(
    base64_string: str,
) -> Tuple[np.ndarray, int]:
    """
    Convert base64 encoded WAV string to audio data and sample rate.

    Args:
        base64_string (str): Base64 encoded WAV data

    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate
    """
    # Decode base64 string
    wav_bytes = base64.b64decode(base64_string)

    # Write to temporary file
    with open("temp.wav", "wb") as f:
        f.write(wav_bytes)

    # Read the WAV file
    audio_data, sample_rate = read_wav_file("temp.wav")

    # Clean up temporary file
    Path("temp.wav").unlink()

    return audio_data, sample_rate
