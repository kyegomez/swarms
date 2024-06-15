**Quickstart**

## Migrate from OpenAI to Swarms in 3 lines of code
If you’ve been using GPT-3.5 or GPT-4, switching to Swarms is easy!

Swarms VLMs are available to use through our OpenAI compatible API. Additionally, if you have been building or prototyping using OpenAI’s Python SDK you can keep your code as-is and use Swarms’s VLMs models.

In this example, we will show you how to change just three lines of code to make your Python application use Swarms’s Open Source models through OpenAI’s Python SDK.

​
## Getting Started
Migrate OpenAI’s Python SDK example script to use Swarms’s LLM endpoints.

These are the three modifications necessary to achieve our goal:

Redefine OPENAI_API_KEY your API key environment variable to use your Swarms key.

Redefine OPENAI_BASE_URL to point to `https://api.swarms.world/v1/chat/completions`

Change the model name to an Open Source model, for example: cogvlm-chat-17b
​
## Requirements
We will be using Python and OpenAI’s Python SDK.
​
## Instructions
Set up a Python virtual environment. Read Creating Virtual Environments here.

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install the pip requirements in your local python virtual environment

`python3 -m pip install openai`
​
## Environment setup
To run this example, there are simple steps to take:

Get an Swarms API token by following these instructions.
Expose the token in a new SWARMS_API_TOKEN environment variable:

`export SWARMS_API_TOKEN=<your-token>`

Switch the OpenAI token and base URL environment variable

`export OPENAI_API_KEY=$SWARMS_API_TOKEN`
`export OPENAI_BASE_URL="https://api.swarms.world/v1/chat/completions"`

If you prefer, you can also directly paste your token into the client initialization.

​
## Example code
Once you’ve completed the steps above, the code below will call Swarms LLMs:

```python
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="cogvlm-chat-17b", 
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)

print(completion.choices[0].message)
``` 

Note that you need to supply one of Swarms’s supported LLMs as an argument, as in the example above. For a complete list of our supported LLMs, check out our REST API page.

​
## Example output
The code above produces the following object:

```
ChatCompletionMessage(content="  Hello! How can I assist you today? Do you have any questions or tasks you'd like help with? Please let me know and I'll do my best to assist you.", role='assistant' function_call=None, tool_calls=None)
```


