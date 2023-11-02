from vllm import LLM, SamplingParams
import openai
import ray
import uvicorn
from vllm.entrypoints import api_server as vllm_api_server
from vllm.entrypoints.openai import api_server as openai_api_server
from skypilot import SkyPilot

class VLLMModel:
    def __init__(self, model_name="facebook/opt-125m", tensor_parallel_size=1):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.model = LLM(model_name, tensor_parallel_size=tensor_parallel_size)
        self.temperature = 1.0
        self.max_tokens = None
        self.sampling_params = SamplingParams(temperature=self.temperature)

    def generate_text(self, prompt: str) -> str:
        output = self.model.generate([prompt], self.sampling_params)
        return output[0].outputs[0].text

    def set_temperature(self, value: float):
        self.temperature = value
        self.sampling_params = SamplingParams(temperature=self.temperature)

    def set_max_tokens(self, value: int):
        self.max_tokens = value
        self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)

    def offline_batched_inference(self, prompts: list) -> list:
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

    def start_api_server(self):
        uvicorn.run(vllm_api_server.app, host="0.0.0.0", port=8000)

    def start_openai_compatible_server(self):
        uvicorn.run(openai_api_server.app, host="0.0.0.0", port=8000)

    def query_openai_compatible_server(self, prompt: str):
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        completion = openai.Completion.create(model=self.model_name, prompt=prompt)
        return completion

    def distributed_inference(self, prompt: str):
        ray.init()
        self.model = LLM(self.model_name, tensor_parallel_size=self.tensor_parallel_size)
        output = self.model.generate(prompt, self.sampling_params)
        ray.shutdown()
        return output[0].outputs[0].text

    def run_on_cloud_with_skypilot(self, yaml_file):
        sky = SkyPilot()
        sky.launch(yaml_file)
