import torch
from PIL import Image
from modelscope import AutoModelForCausalLM, AutoTokenizer
from swarms.models.base_multimodal_model import BaseMultiModalModel

device_check = "cuda" if torch.cuda.is_available() else "cpu"


class CogAgent(BaseMultiModalModel):
    """CogAgent
    
    Multi-modal conversational agent that can be used to chat with
    images and text. It is based on the CogAgent model from the
    ModelScope library.
    
    Attributes:
        model_name (str): The name of the model to be used
        tokenizer_name (str): The name of the tokenizer to be used
        dtype (torch.bfloat16): The data type to be used
        low_cpu_mem_usage (bool): Whether to use low CPU memory
        load_in_4bit (bool): Whether to load in 4-bit
        trust_remote_code (bool): Whether to trust remote code
        device (str): The device to be used
    
    Examples:
        >>> from swarms.models.cog_agent import CogAgent
        >>> cog_agent = CogAgent()
        >>> cog_agent.run("How are you?", "images/1.jpg")
        <s> I'm fine. How are you? </s>
    """
    def __init__(
        self,
        model_name: str = "ZhipuAI/cogagent-chat",
        tokenizer_name: str = "I-ModelScope/vicuna-7b-v1.5",
        dtype=torch.bfloat16,
        low_cpu_mem_usage: bool = True,
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
        device=device_check,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.dtype = dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.device = device

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                load_in_4bit=self.load_in_4bit,
                trust_remote_code=self.trust_remote_code,
                *args,
                **kwargs,
            )
            .to(self.device)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name
        )

    def run(self, task: str, img: str, *args, **kwargs):
        """Run the model

        Args:
            task (str): The task to be performed
            img (str): The image path
            
        """ 
        image = Image.open(img).convert("RGB")

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=task,
            history=[],
            images=[image],
        )

        inputs = {
            "input_ids": (
                input_by_model["input_ids"]
                .unsqueeze(0)
                .to(self.device)
            ),
            "token_type_ids": (
                input_by_model["token_type_ids"]
                .unsqueeze(0)
                .to(self.device)
            ),
            "attention_mask": (
                input_by_model["attention_mask"]
                .unsqueeze(0)
                .to(self.device)
            ),
            "images": [
                [
                    input_by_model["images"][0]
                    .to(self.device)
                    .to(self.dtype)
                ]
            ],
        }
        if (
            "cross_images" in input_by_model
            and input_by_model["cross_images"]
        ):
            inputs["cross_images"] = [
                [
                    input_by_model["cross_images"][0]
                    .to(self.device)
                    .to(self.dtype)
                ]
            ]

        with torch.no_grad():
            outputs = self.model(**inputs, **kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.decode(outputs[0])
            response = response.split("</s>")[0]
            print(response)
