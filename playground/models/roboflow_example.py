from swarms import RoboflowMultiModal

# Initialize the model
model = RoboflowMultiModal(
    api_key="api",
    project_id="your project id",
    hosted=False,
)


# Run the model on an img
out = model("img.png")
