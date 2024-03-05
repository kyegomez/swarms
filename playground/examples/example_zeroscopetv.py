# Import the model
from swarms import ZeroscopeTTV

# Initialize the model
zeroscope = ZeroscopeTTV()

# Specify the task
task = "A person is walking on the street."

# Generate the video!
video_path = zeroscope(task)
print(video_path)
