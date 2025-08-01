from litellm import model_list

for model in model_list:
    if "azure" in model:
        print(model)
