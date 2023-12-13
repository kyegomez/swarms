from swarms.models import idefics

model = idefics()

user_input = (
    "User: What is in this image?"
    " https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
)
response = model.chat(user_input)
print(response)

user_input = (
    "User: And who is that?"
    " https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
)
response = model.chat(user_input)
print(response)

model.set_checkpoint("new_checkpoint")
model.set_device("cpu")
model.set_max_length(200)
model.clear_chat_history()
