from swarms.models.vilt import Vilt

model = Vilt()

output = model(
    "What is this image",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
)
