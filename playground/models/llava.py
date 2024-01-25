from swarms import QwenVLMultiModal

# Instantiate the QwenVLMultiModal model
model = QwenVLMultiModal(
    model_name="Qwen/Qwen-VL-Chat",
    device="cuda",
    quantize=True,
)

# Run the model
response = model(
    "Hello, how are you?", "https://example.com/image.jpg"
)

# Print the response
print(response)
