# Import the idefics model from the swarms.models module
from swarms.models import Idefics

# Create an instance of the idefics model
model = Idefics()

# Define user input with an image URL and chat with the model
user_input = (
    "User: What is in this image?"
    " https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
)
response = model.chat(user_input)
print(response)

# Define another user input with an image URL and chat with the model
user_input = (
    "User: And who is that?"
    " https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
)
response = model.chat(user_input)
print(response)

# Set the checkpoint of the model to "new_checkpoint"
model.set_checkpoint("new_checkpoint")

# Set the device of the model to "cpu"
model.set_device("cpu")

# Set the maximum length of the chat to 200
model.set_max_length(200)

# Clear the chat history of the model
model.clear_chat_history()
