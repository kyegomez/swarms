from swarms.models import Mistral

model = Mistral(device="cuda", use_flash_attention=True)

prompt = "My favourite condiment is"
result = model.run(prompt)
print(result)
