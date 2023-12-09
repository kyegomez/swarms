# Query Wolframalpha Service

Contributor [Shengding Hu](https://github.com/shengdinghu)

You can get the API keys from https://products.wolframalpha.com/api/

# Wolfram Tool

This tool provides dynamic computation and curated data from WolframAlpha and Wolfram Cloud.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Wolfram"
- **description_for_model**: "Dynamic computation and curated data from WolframAlpha and Wolfram Cloud."
- **logo_url**: "https://www.wolframcdn.com/images/icons/Wolfram.png"
- **contact_email**: "hello@contact.com"
- **legal_info_url**: "hello@legal.com"

## Endpoint

The tool provides the following endpoint:

- **/getWolframAlphaResults**: Get Wolfram|Alpha results using a natural query.

## Function Description

- **getWolframAlphaResults(input: str) -> dict**: This function gets Wolfram|Alpha results using a natural query. The input should be a string. The function returns a dictionary containing the results. Note that queries to getWolframAlphaResults must ALWAYS have this structure: {"input": query}. Please directly read the output JSON.