## ChatGPT User Guide with Abstraction

Welcome to the ChatGPT user guide! This document will walk you through the Reverse Engineered ChatGPT API, its usage, and how to leverage the abstraction in `revgpt.py` for seamless integration.

### Table of Contents
1. [Installation](#installation)
2. [Initial Setup and Configuration](#initial-setup)
3. [Using the Abstract Class from `revgpt.py`](#using-abstract-class)
4. [V1 Standard ChatGPT](#v1-standard-chatgpt)
5. [V3 Official Chat API](#v3-official-chat-api)
6. [Credits & Disclaimers](#credits-disclaimers)

### Installation <a name="installation"></a>

To kickstart your journey with ChatGPT, first, install the ChatGPT package:

```shell
python -m pip install --upgrade revChatGPT
```

**Supported Python Versions:**
- Minimum: Python3.10
- Recommended: Python3.11+

### Initial Setup and Configuration <a name="initial-setup"></a>

1. **Account Setup:** Register on [OpenAI's ChatGPT](https://chat.openai.com/).
2. **Authentication:** Obtain your access token from OpenAI's platform.
3. **Environment Variables:** Configure your environment with the necessary variables. An example of these variables can be found at the bottom of the guide.

### Using the Abstract Class from `revgpt.py` <a name="using-abstract-class"></a>

The abstraction provided in `revgpt.py` is designed to simplify your interactions with ChatGPT.

1. **Import the Necessary Modules:**

```python
from dotenv import load_dotenv
from revgpt import AbstractChatGPT
```

2. **Load Environment Variables:**

```python
load_dotenv()
```

3. **Initialize the ChatGPT Abstract Class:**

```python
chat = AbstractChatGPT(api_key=os.getenv("ACCESS_TOKEN"), **config)
```

4. **Start Interacting with ChatGPT:**

```python
response = chat.ask("Hello, ChatGPT!")
print(response)
```

With the abstract class, you can seamlessly switch between different versions or models of ChatGPT without changing much of your code.

### V1 Standard ChatGPT <a name="v1-standard-chatgpt"></a>

If you wish to use V1 specifically:

1. Import the model:

```python
from swarms.models.revgptV1 import RevChatGPTModelv1
```

2. Initialize:

```python
model = RevChatGPTModelv1(access_token=os.getenv("ACCESS_TOKEN"), **config)
```

3. Interact:

```python
response = model.run("What's the weather like?")
print(response)
```

### V3 Official Chat API <a name="v3-official-chat-api"></a>

For users looking to integrate the official V3 API:

1. Import the model:

```python
from swarms.models.revgptV4 import RevChatGPTModelv4
```

2. Initialize:

```python
model = RevChatGPTModelv4(access_token=os.getenv("OPENAI_API_KEY"), **config)
```

3. Interact:

```python
response = model.run("Tell me a fun fact!")
print(response)
```

### Credits & Disclaimers <a name="credits-disclaimers"></a>

- This project is not an official OpenAI product and is not affiliated with OpenAI. Use at your own discretion.
- Many thanks to all the contributors who have made this project possible.
- Special acknowledgment to [virtualharby](https://www.youtube.com/@virtualharby) for the motivating music!

---

By following this guide, you should now have a clear understanding of how to use the Reverse Engineered ChatGPT API and its abstraction. Happy coding!
