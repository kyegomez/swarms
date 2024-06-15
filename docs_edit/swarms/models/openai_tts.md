# `OpenAITTS` Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Initialization](#initialization)
   - [Running TTS](#running-tts)
   - [Running TTS and Saving](#running-tts-and-saving)
4. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Saving the Output](#saving-the-output)
5. [Advanced Options](#advanced-options)
6. [Troubleshooting](#troubleshooting)
7. [References](#references)

## 1. Overview <a name="overview"></a>

The `OpenAITTS` module is a Python library that provides an interface for converting text to speech (TTS) using the OpenAI TTS API. It allows you to generate high-quality speech from text input, making it suitable for various applications such as voice assistants, speech synthesis, and more.

### Features:
- Convert text to speech using OpenAI's TTS model.
- Supports specifying the model name, voice, and other parameters.
- Option to save the generated speech to a WAV file.

## 2. Installation <a name="installation"></a>

To use the `OpenAITTS` model, you need to install the necessary dependencies. You can do this using `pip`:

```bash
pip install swarms requests wave
```

## 3. Usage <a name="usage"></a>

### Initialization <a name="initialization"></a>

To use the `OpenAITTS` module, you need to initialize an instance of the `OpenAITTS` class. Here's how you can do it:

```python
from swarms.models.openai_tts import OpenAITTS

# Initialize the OpenAITTS instance
tts = OpenAITTS(
    model_name="tts-1-1106",
    proxy_url="https://api.openai.com/v1/audio/speech",
    openai_api_key=openai_api_key_env,
    voice="onyx",
)
```

#### Parameters:
- `model_name` (str): The name of the TTS model to use (default is "tts-1-1106").
- `proxy_url` (str): The URL for the OpenAI TTS API (default is "https://api.openai.com/v1/audio/speech").
- `openai_api_key` (str): Your OpenAI API key. It can be obtained from the OpenAI website.
- `voice` (str): The voice to use for generating speech (default is "onyx").
- `chunk_size` (int): The size of data chunks when fetching audio (default is 1024 * 1024 bytes).
- `autosave` (bool): Whether to automatically save the generated speech to a file (default is False).
- `saved_filepath` (str): The path to the file where the speech will be saved (default is "runs/tts_speech.wav").

### Running TTS <a name="running-tts"></a>

Once the `OpenAITTS` instance is initialized, you can use it to convert text to speech using the `run` method:

```python
# Generate speech from text
speech_data = tts.run("Hello, world!")
```

#### Parameters:
- `task` (str): The text you want to convert to speech.

#### Returns:
- `speech_data` (bytes): The generated speech data.

### Running TTS and Saving <a name="running-tts-and-saving"></a>

You can also use the `run_and_save` method to generate speech from text and save it to a file:

```python
# Generate speech from text and save it to a file
speech_data = tts.run_and_save("Hello, world!")
```

#### Parameters:
- `task` (str): The text you want to convert to speech.

#### Returns:
- `speech_data` (bytes): The generated speech data.

## 4. Examples <a name="examples"></a>

### Basic Usage <a name="basic-usage"></a>

Here's a basic example of how to use the `OpenAITTS` module to generate speech from text:

```python
from swarms.models.openai_tts import OpenAITTS

# Initialize the OpenAITTS instance
tts = OpenAITTS(
    model_name="tts-1-1106",
    proxy_url="https://api.openai.com/v1/audio/speech",
    openai_api_key=openai_api_key_env,
    voice="onyx",
)

# Generate speech from text
speech_data = tts.run("Hello, world!")
```

### Saving the Output <a name="saving-the-output"></a>

You can save the generated speech to a WAV file using the `run_and_save` method:

```python
# Generate speech from text and save it to a file
speech_data = tts.run_and_save("Hello, world!")
```

## 5. Advanced Options <a name="advanced-options"></a>

The `OpenAITTS` module supports various advanced options for customizing the TTS generation process. You can specify the model name, voice, and other parameters during initialization. Additionally, you can configure the chunk size for audio data fetching and choose whether to automatically save the generated speech to a file.

## 6. Troubleshooting <a name="troubleshooting"></a>

If you encounter any issues while using the `OpenAITTS` module, please make sure you have installed all the required dependencies and that your OpenAI API key is correctly configured. If you still face problems, refer to the OpenAI documentation or contact their support for assistance.

## 7. References <a name="references"></a>

- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Python Requests Library](https://docs.python-requests.org/en/latest/)
- [Python Wave Library](https://docs.python.org/3/library/wave.html)

This documentation provides a comprehensive guide on how to use the `OpenAITTS` module to convert text to speech using OpenAI's TTS model. It covers initialization, basic usage, advanced options, troubleshooting, and references for further exploration.