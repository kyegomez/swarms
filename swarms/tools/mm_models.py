import os
import uuid

import numpy as np
import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)

from swarms.models.prompts.prebuild.multi_modal_prompts import IMAGE_PROMPT
from swarms.tools.base import tool
from swarms.tools.main import BaseToolSet
from swarms.utils.logger import logger
from swarms.utils.main import BaseHandler, get_new_image_name


class MaskFormer(BaseToolSet):
    def __init__(self, device):
        print("Initializing MaskFormer to %s" % device)
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(
            text=text, images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(original_image.size)


class ImageEditing(BaseToolSet):
    def __init__(self, device):
        print("Initializing ImageEditing to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.revision = "fp16" if "cuda" in device else None
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision=self.revision,
            torch_dtype=self.torch_dtype,
        ).to(device)

    @tool(
        name="Remove Something From The Photo",
        description="useful when you want to remove and object or something from the photo "
        "from its description or location. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the object need to be removed. ",
    )
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")
        return self.inference_replace(f"{image_path},{to_be_removed_txt},background")

    @tool(
        name="Replace Something From The Photo",
        description="useful when you want to replace an object from the object description or "
        "location with another object from its description. "
        "The input to this tool should be a comma separated string of three, "
        "representing the image_path, the object to be replaced, the object to be replaced with ",
    )
    def inference_replace(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpaint(
            prompt=replace_with_txt,
            image=original_image.resize((512, 512)),
            mask_image=mask_image.resize((512, 512)),
        ).images[0]
        updated_image_path = get_new_image_name(
            image_path, func_name="replace-something"
        )
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)

        logger.debug(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}"
        )

        return updated_image_path


class InstructPix2Pix(BaseToolSet):
    def __init__(self, device):
        print("Initializing InstructPix2Pix to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    @tool(
        name="Instruct Image Using Text",
        description="useful when you want to the style of the image to be like the text. "
        "like: make it look like a painting. or make it like a robot. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the text. ",
    )
    def inference(self, inputs):
        """Change style of image."""
        logger.debug("===> Starting InstructPix2Pix Inference")
        image_path, text = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        original_image = Image.open(image_path)
        image = self.pipe(
            text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)

        logger.debug(
            f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
            f"Output Image: {updated_image_path}"
        )

        return updated_image_path


class Text2Image(BaseToolSet):
    def __init__(self, device):
        print("Initializing Text2Image to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=self.torch_dtype
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @tool(
        name="Generate Image From User Input Text",
        description="useful when you want to generate an image from a user input text and save it to a file. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, text):
        image_filename = os.path.join("image", str(uuid.uuid4())[0:8] + ".png")
        prompt = text + ", " + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)

        logger.debug(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}"
        )

        return image_filename


class VisualQuestionAnswering(BaseToolSet):
    def __init__(self, device):
        print("Initializing VisualQuestionAnswering to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    @tool(
        name="Answer Question About The Image",
        description="useful when you need an answer for a question based on an image. "
        "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
        "The input to this tool should be a comma separated string of two, representing the image_path and the question",
    )
    def inference(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(raw_image, question, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)

        logger.debug(
            f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
            f"Output Answer: {answer}"
        )

        return answer
    


class ImageCaptioning(BaseHandler):
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    def handle(self, filename: str):
        img = Image.open(filename)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert("RGB")
        img.save(filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")

        inputs = self.processor(Image.open(filename), return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        description = self.processor.decode(out[0], skip_special_tokens=True)
        print(
            f"\nProcessed ImageCaptioning, Input Image: {filename}, Output Text: {description}"
        )

        return IMAGE_PROMPT.format(filename=filename, description=description)
    




