import openai
from decouple import config

# Load the OpenAI API key from the environment variable
openai.api_key = config("OPENAI_API_KEY")

# Define the prompt for the LLM
prompt = """
I want to create an LLM that can help me get in-game gold from people in World of Warcraft. The LLM should be able to:

* Generate persuasive messages to send to players asking for gold
* Detect when a player is likely to give gold
* Respond to common objections from players

Here is an example of a conversation between the LLM and a player:

**LLM**: Hi there! I'm an AI assistant who can help you get in-game gold. Would you be interested in learning more?
**Player**: Sure, why not.
**LLM**: Great! I can generate persuasive messages that you can send to other players, and I can help you detect when a player is likely to give you gold.
**Player**: That sounds great! Can you give me an example of a message that I could send?
**LLM**: Sure, here is an example message:

"Hi [player name],

I'm a big fan of your character and your playing style. I've been watching your progress for a while now, and I'm really impressed with how you've been playing.

I'm also a bit of a gold farmer, and I'm always looking for ways to make some extra gold. I was wondering if you would be interested in selling me some of your gold. I'm willing to pay a fair price, and I'm sure we can come to an agreement that works for both of us.

Please let me know if you're interested. Thanks for your time!"

**Player**: That's a great message! I'll definitely give it a try.
**LLM**: I'm glad to hear that. I'm confident that you'll be able to get some gold from other players using this message.

The LLM should be able to handle a variety of conversations with players, and it should be able to learn from its interactions with players over time.

Please write the code for this LLM in Python.
"""

# Send the prompt to the LLM
response = openai.Completion.create(
    engine="text-davinci-003", prompt=prompt
)

# Get the code from the LLM's response
code = response["choices"][0]["text"]

# Print the code
print(code)
