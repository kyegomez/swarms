from swarms.models.kosmos2 import Kosmos2
from PIL import Image


model = Kosmos2.initialize()

image = Image.open("images/swarms.jpg")

detections = model(image)
print(detections)
