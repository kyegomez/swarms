# ElevenLabsText2SpeechTool Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Class Overview](#class-overview)
   - [Attributes](#attributes)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Initialization](#initialization)
   - [Converting Text to Speech](#converting-text-to-speech)
   - [Playing and Streaming Speech](#playing-and-streaming-speech)
5. [Exception Handling](#exception-handling)
6. [Advanced Usage](#advanced-usage)
7. [Contributing](#contributing)
8. [References](#references)

## 1. Introduction <a name="introduction"></a>
The `ElevenLabsText2SpeechTool` is a Python class designed to simplify the process of converting text to speech using the Eleven Labs Text2Speech API. This tool is a wrapper around the API and provides a convenient interface for generating speech from text. It supports multiple languages, making it suitable for a wide range of applications, including voice assistants, audio content generation, and more.

## 2. Class Overview <a name="class-overview"></a>
### Attributes <a name="attributes"></a>
- `model` (Union[ElevenLabsModel, str]): The model to use for text to speech. Defaults to `ElevenLabsModel.MULTI_LINGUAL`.
- `name` (str): The name of the tool. Defaults to `"eleven_labs_text2speech"`.
- `description` (str): A brief description of the tool. Defaults to a detailed explanation of its functionality.

## 3. Installation <a name="installation"></a>
To use the `ElevenLabsText2SpeechTool`, you need to install the required dependencies and have access to the Eleven Labs Text2Speech API. Follow these steps:

1. Install the `elevenlabs` library:
   ```
   pip install elevenlabs
   ```

2. Install the `swarms` library
    `pip install swarms`

3. Set up your API key by following the instructions at [Eleven Labs Documentation](https://docs.elevenlabs.io/welcome/introduction).

## 4. Usage <a name="usage"></a>
### Initialization <a name="initialization"></a>
To get started, create an instance of the `ElevenLabsText2SpeechTool`. You can customize the `model` attribute if needed.

```python
from swarms.models import ElevenLabsText2SpeechTool

stt = ElevenLabsText2SpeechTool(model=ElevenLabsModel.MONO_LINGUAL)
```

### Converting Text to Speech <a name="converting-text-to-speech"></a>
You can use the `run` method to convert text to speech. It returns the path to the generated speech file.

```python
speech_file = stt.run("Hello, this is a test.")
```

### Playing and Streaming Speech <a name="playing-and-streaming-speech"></a>
- Use the `play` method to play the generated speech file.

```python
stt.play(speech_file)
```

- Use the `stream_speech` method to stream the text as speech. It plays the speech in real-time.

```python
stt.stream_speech("Hello world!")
```

## 5. Exception Handling <a name="exception-handling"></a>
The `ElevenLabsText2SpeechTool` handles exceptions gracefully. If an error occurs during the conversion process, it raises a `RuntimeError` with an informative error message.

## 6. Advanced Usage <a name="advanced-usage"></a>
- You can implement custom error handling and logging to further enhance the functionality of this tool.
- For advanced users, extending the class to support additional features or customization is possible.

## 7. Contributing <a name="contributing"></a>
Contributions to this tool are welcome. Feel free to open issues, submit pull requests, or provide feedback to improve its functionality and documentation.

## 8. References <a name="references"></a>
- [Eleven Labs Text2Speech API Documentation](https://docs.elevenlabs.io/welcome/introduction)

This documentation provides a comprehensive guide to using the `ElevenLabsText2SpeechTool`. It covers installation, basic usage, advanced features, and contribution guidelines. Refer to the [References](#references) section for additional resources.