import gradio_client as grc
import interpreter
import time
import gradio as gr
from pydub import AudioSegment
import io
from elevenlabs import generate, play, set_api_key
import whisper
import dotenv

dotenv.load_dotenv(".env")

# interpreter.model = "TheBloke/Mistral-7B-OpenOrca-GGUF"
interpreter.auto_run = True
model = whisper.load_model("base")


def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text


set_api_key("ELEVEN_LABS_API_KEY")


def get_audio_length(audio_bytes):
    # Create a BytesIO object from the byte array
    byte_io = io.BytesIO(audio_bytes)

    # Load the audio data with PyDub
    audio = AudioSegment.from_mp3(byte_io)

    # Get the length of the audio in milliseconds
    length_ms = len(audio)

    # Optionally convert to seconds
    length_s = length_ms / 1000.0

    return length_s


def speak(text):
    speaking = True
    audio = generate(
        text=text,
        voice="Daniel"
    )
    play(audio, notebook=True)

    audio_length = get_audio_length(audio)
    time.sleep(audio_length)

# @title Text-only JARVIS
# @markdown Run this cell for a ChatGPT-like interface.


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):

        user_message = history[-1][0]
        history[-1][1] = ""
        active_block_type = ""

        for chunk in interpreter.chat(user_message, stream=True, display=False):

            # Message
            if "message" in chunk:
                if active_block_type != "message":
                    active_block_type = "message"
                history[-1][1] += chunk["message"]
                yield history

            # Code
            if "language" in chunk:
                language = chunk["language"]
            if "code" in chunk:
                if active_block_type != "code":
                    active_block_type = "code"
                    history[-1][1] += f"\n```{language}\n"
                history[-1][1] += chunk["code"]
                yield history

            # Output
            if "executing" in chunk:
                history[-1][1] += "\n```\n\n```text\n"
                yield history
            if "output" in chunk:
                if chunk["output"] != "KeyboardInterrupt":
                    history[-1][1] += chunk["output"] + "\n"
                    yield history
            if "end_of_execution" in chunk:
                history[-1][1] = history[-1][1].strip()
                history[-1][1] += "\n```\n"
                yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

if __name__ == '__main__':
    demo.queue()
    demo.launch(debug=True)
