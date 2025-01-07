from __future__ import annotations

import asyncio
import base64
import io
import threading
from collections.abc import Awaitable
from os import getenv
from typing import Any, Callable, cast

import numpy as np

try:
    import pyaudio
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "pyaudio"])
    import pyaudio
try:
    import sounddevice as sd
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "sounddevice"])
    import sounddevice as sd
from loguru import logger
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import (
    AsyncRealtimeConnection,
)
from openai.types.beta.realtime.session import Session

try:
    from pydub import AudioSegment
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "pydub"])
    from pydub import AudioSegment

from dotenv import load_dotenv

load_dotenv()


CHUNK_LENGTH_S = 0.05  # 100ms
SAMPLE_RATE = 24000
FORMAT = pyaudio.paInt16
CHANNELS = 1

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false


def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
    # load the audio file from the byte stream
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    print(
        f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}"
    )
    # resample to 24kHz mono pcm16
    pcm_audio = (
        audio.set_frame_rate(SAMPLE_RATE)
        .set_channels(CHANNELS)
        .set_sample_width(2)
        .raw_data
    )
    return pcm_audio


class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
            blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
        )
        self.playing = False
        self._frame_count = 0

    def callback(self, outdata, frames, time, status):
        with self.lock:
            data = np.empty(0, dtype=np.int16)

            # get next item from queue if there is still space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # fill the rest of the frames with zeros if there is no more data
            if len(data) < frames:
                data = np.concatenate(
                    (
                        data,
                        np.zeros(frames - len(data), dtype=np.int16),
                    )
                )

        outdata[:] = data.reshape(-1, 1)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        with self.lock:
            # bytes is pcm16 single channel audio data, convert to numpy array
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        self.playing = True
        self.stream.start()

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []

    def terminate(self):
        self.stream.close()


async def send_audio_worker_sounddevice(
    connection: AsyncRealtimeConnection,
    should_send: Callable[[], bool] | None = None,
    start_send: Callable[[], Awaitable[None]] | None = None,
):
    sent_audio = False

    device_info = sd.query_devices()
    print(device_info)

    read_size = int(SAMPLE_RATE * 0.02)

    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
    )
    stream.start()

    try:
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue

            data, _ = stream.read(read_size)

            if should_send() if should_send else True:
                if not sent_audio and start_send:
                    await start_send()
                await connection.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode(
                            "utf-8"
                        ),
                    }
                )
                sent_audio = True

            elif sent_audio:
                print("Done, triggering inference")
                await connection.send(
                    {"type": "input_audio_buffer.commit"}
                )
                await connection.send(
                    {"type": "response.create", "response": {}}
                )
                sent_audio = False

            await asyncio.sleep(0)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


class RealtimeApp:
    """
    A console-based application to handle real-time audio recording and streaming,
    connecting to OpenAI's GPT-4 Realtime API.

    Features:
        - Streams microphone input to the GPT-4 Realtime API.
        - Logs transcription results.
        - Sends text prompts to the GPT-4 Realtime API.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self.connection: AsyncRealtimeConnection | None = None
        self.session: Session | None = None
        self.client = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"))
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id: str | None = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.system_prompt = system_prompt

    async def initialize_text_prompt(self, text: str) -> None:
        """Initialize and send a text prompt to the OpenAI Realtime API."""
        try:
            async with self.client.beta.realtime.connect(
                model="gpt-4o-realtime-preview-2024-10-01"
            ) as conn:
                self.connection = conn
                await conn.session.update(
                    session={"modalities": ["text"]}
                )

                await conn.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": text}
                        ],
                    }
                )
                await conn.response.create()

                async for event in conn:
                    if event.type == "response.text.delta":
                        print(event.delta, flush=True, end="")

                    elif event.type == "response.text.done":
                        print()

                    elif event.type == "response.done":
                        break
        except Exception as e:
            logger.exception(f"Error initializing text prompt: {e}")

    async def handle_realtime_connection(self) -> None:
        """Handle the connection to the OpenAI Realtime API."""
        try:
            async with self.client.beta.realtime.connect(
                model="gpt-4o-realtime-preview-2024-10-01"
            ) as conn:
                self.connection = conn
                self.connected.set()
                logger.info("Connected to OpenAI Realtime API.")

                await conn.session.update(
                    session={"turn_detection": {"type": "server_vad"}}
                )

                acc_items: dict[str, Any] = {}

                async for event in conn:
                    if event.type == "session.created":
                        self.session = event.session
                        assert event.session.id is not None
                        logger.info(
                            f"Session created with ID: {event.session.id}"
                        )
                        continue

                    if event.type == "session.updated":
                        self.session = event.session
                        logger.info("Session updated.")
                        continue

                    if event.type == "response.audio.delta":
                        if event.item_id != self.last_audio_item_id:
                            self.audio_player.reset_frame_count()
                            self.last_audio_item_id = event.item_id

                        bytes_data = base64.b64decode(event.delta)
                        self.audio_player.add_data(bytes_data)
                        continue

                    if (
                        event.type
                        == "response.audio_transcript.delta"
                    ):
                        try:
                            text = acc_items[event.item_id]
                        except KeyError:
                            acc_items[event.item_id] = event.delta
                        else:
                            acc_items[event.item_id] = (
                                text + event.delta
                            )

                        logger.debug(
                            f"Transcription updated: {acc_items[event.item_id]}"
                        )
                        continue

                    if event.type == "response.text.delta":
                        print(event.delta, flush=True, end="")
                        continue

                    if event.type == "response.text.done":
                        print()
                        continue

                    if event.type == "response.done":
                        break
        except Exception as e:
            logger.exception(
                f"Error in realtime connection handler: {e}"
            )

    async def _get_connection(self) -> AsyncRealtimeConnection:
        """Wait for and return the realtime connection."""
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_text_prompt(self, text: str) -> None:
        """Send a text prompt to the OpenAI Realtime API."""
        try:
            connection = await self._get_connection()
            if not self.session:
                logger.error(
                    "Session is not initialized. Cannot send prompt."
                )
                return

            logger.info(f"Sending prompt to the model: {text}")
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            )
            await connection.response.create()
        except Exception as e:
            logger.exception(f"Error sending text prompt: {e}")

    async def send_mic_audio(self) -> None:
        """Stream microphone audio to the OpenAI Realtime API."""
        import sounddevice as sd  # type: ignore

        sent_audio = False

        try:
            read_size = int(SAMPLE_RATE * 0.02)
            stream = sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="int16",
            )
            stream.start()

            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()

                data, _ = stream.read(read_size)

                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(
                        connection.send({"type": "response.cancel"})
                    )
                    sent_audio = True

                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(cast(Any, data)).decode(
                        "utf-8"
                    )
                )
                await asyncio.sleep(0)
        except Exception as e:
            logger.exception(
                f"Error in microphone audio streaming: {e}"
            )
        finally:
            stream.stop()
            stream.close()

    async def run(self) -> None:
        """Start the application tasks."""
        logger.info("Starting application tasks.")

        await asyncio.gather(
            # self.initialize_text_prompt(self.system_prompt),
            self.handle_realtime_connection(),
            self.send_mic_audio(),
        )


if __name__ == "__main__":
    logger.add(
        "realtime_app.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
    )
    logger.info("Starting RealtimeApp.")
    app = RealtimeApp()
    asyncio.run(app.run())
