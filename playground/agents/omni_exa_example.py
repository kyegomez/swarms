# pip3 install exxa
from exa import Inference
from swarms.agents import OmniModalAgent

llm = Inference(model_id="mistralai/Mistral-7B-v0.1", quantize=True)

agent = OmniModalAgent(llm)

agent.run("Create a video of a swarm of fish")
