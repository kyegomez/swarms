from swarms.models import Dalle3

dalle3 = Dalle3(openai_api_key="", openai_org_id=""))
task = "A painting of a dog"
image_url = dalle3(task)
print(image_url)
