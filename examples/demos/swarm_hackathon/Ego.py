import os
import random

# Create a list of character names
character_names = ["Alice", "Bob", "Charlie", "Dave", "Eve"]

# Create a dictionary of character voices
character_voices = {
    "Alice": "Alice.wav",
    "Bob": "Bob.wav",
    "Charlie": "Charlie.wav",
    "Dave": "Dave.wav",
    "Eve": "Eve.wav",
}

# Get the user's input
conversation_topic = input(
    "What would you like the characters to talk about? "
)


# Create a function to generate a random conversation
def generate_conversation(characters, topic):
    # Choose two random characters to talk
    character1 = random.choice(characters)
    character2 = random.choice(characters)

    # Generate the conversation
    conversation = [
        (
            f"{character1}: Hello, {character2}. I'd like to talk"
            f" about {topic}."
        ),
        (
            f"{character2}: Sure, {character1}. What do you want to"
            " know?"
        ),
        (
            f"{character1}: I'm just curious about your thoughts on"
            " the matter."
        ),
        f"{character2}: Well, I think it's a very interesting topic.",
        f"{character1}: I agree. I'm glad we're talking about this.",
    ]

    # Return the conversation
    return conversation


# Generate the conversation
conversation = generate_conversation(
    character_names, conversation_topic
)

# Play the conversation
for line in conversation:
    print(line)
    os.system(f"afplay {character_voices[line.split(':')[0]]}")
