from swarms.models.ssd_1b import SSD1B

model = SSD1B()

task = "A painting of a dog"
neg_prompt = "ugly, blurry, poor quality"

image_url = model(task, neg_prompt)
print(image_url)
