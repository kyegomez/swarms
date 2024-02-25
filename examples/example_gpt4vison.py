from swarms import GPT4VisionAPI

# Initialize with default API key and custom max_tokens
api = GPT4VisionAPI(max_tokens=1000)

# Define the task and image URL
task = "Describe the scene in the image."
img = "/home/kye/.swarms/swarms/examples/Screenshot from 2024-02-20 05-55-34.png"

# Run the GPT-4 Vision model
response = api.run(task, img)

# Print the model's response
print(response)
