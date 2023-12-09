# Translation Queries

Contributor: [Shihao Liang](https://github.com/pooruss)

# NLLB as translation tool:
- Language code should be included in queries in case the NLLB model cannot recognize the target language.
- Default source language is English. Change the nllb model path and source language in host_local_tools.py by passing a dict like this: {"model": ${your_nllb_path}, "src_lang": "eng_Latn"}
- Refer to paper [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) for all supported languages.

## Translator Info Tool

This tool allows you to translate a given text from one language to another.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Translator Info"
- **description_for_model**: "Plugin for translating text from one language to another."
- **logo_url**: "https://your-app-url.com/.well-known/logo.png"
- **contact_email**: "shihaoliang0828@gmail.com"
- **legal_info_url**: "hello@legal.com"

## Model

The tool uses a model for translation. The base model is "facebook/nllb-200-distilled-600M" and the source language is "eng_Latn" by default. You can change these parameters in the config.

## Endpoint

The tool provides the following endpoint:

- **/get_translation**: Translate a given text from one language to another. The input should be a text string or a list of text strings, the target language, and the maximum length of the translated text.

## Function Description

- **get_translation(input_text: str or list, tgt_lang: str, max_length: int) -> str or list**: This function translates a given text from one language to another. The input_text should be a string or a list of strings, tgt_lang should be a string representing the language code, and max_length should be an integer. The function returns the translated text or a list of translated texts.


# BAIDU translate api as translation tool:
- Set your APP ID and Secert Key in secret_keys.sh.
- Language code should also be included in queries.
- Refer to [Baidu translate](https://fanyi-api.baidu.com/product/11) for all supported languages.

##  Translator Info Tool with Baidu

This tool allows you to translate a given text from one language to another.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Translator Info"
- **description_for_model**: "Plugin for translating text from one language to another."
- **logo_url**: "https://your-app-url.com/.well-known/logo.png"
- **contact_email**: "shihaoliang0828@gmail.com"
- **legal_info_url**: "hello@legal.com"

## API Key

The tool requires an API key and a secret key from Baidu. You can sign up for a free account at https://fanyi-api.baidu.com/, create a new API key and secret key, and add them to environment variables.

## Endpoint

The tool provides the following endpoint:

- **/get_translation**: Translate a given text from one language to another. The input should be a text string and the target language.

## Function Description

- **get_translation(text: str, tgt_lang: str) -> str**: This function translates a given text from one language to another. The text should be a string and the target language should be a string representing the language code. The function returns the translated text.