from swarms.models import HuggingfaceLLM

model_id = "NousResearch/Yarn-Mistral-7b-128k"
inference = HuggingfaceLLM(model_id=model_id)

task = "Once upon a time"
generated_text = inference(task)
print(generated_text)
