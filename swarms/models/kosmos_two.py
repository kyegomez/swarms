import os

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from swarms.models.base_multimodal_model import BaseMultiModalModel


# utils
def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


class Kosmos(BaseMultiModalModel):
    """
    Kosmos model by Yen-Chun Shieh

    Parameters
    ----------
    model_name : str
        Path to the pretrained model

    Examples
    --------
    >>> kosmos = Kosmos()
    >>> kosmos("Hello, my name is", "path/to/image.png")

    """

    def __init__(
        self,
        model_name="ydshieh/kosmos-2-patch14-224",
        max_new_tokens: int = 64,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.max_new_tokens = max_new_tokens

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, trust_remote_code=True, *args, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, *args, **kwargs
        )

    def get_image(self, url: str):
        """Get image from url

        Args:
            url (str): url of image

        Returns:
            _type_: _description_
        """
        return Image.open(requests.get(url, stream=True).raw)

    def run(self, task: str, image: str, *args, **kwargs):
        """Run the model

        Args:
            task (str): task to run
            image (str): img url
        """
        inputs = self.processor(
            text=task, images=image, return_tensors="pt"
        )
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"][:, :-1],
            attention_mask=inputs["attention_mask"][:, :-1],
            image_embeds=None,
            img_attn_mask=inputs["img_attn_mask"][:, :-1],
            use_cache=True,
            max_new_tokens=self.max_new_tokens,
        )

        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        (
            processed_text,
            entities,
        ) = self.processor.post_process_generation(generated_texts)

        return processed_text, entities

    # tasks
    def multimodal_grounding(self, phrase, image_url):
        task = f"<grounding><phrase> {phrase} </phrase>"
        self.run(task, image_url)

    def referring_expression_comprehension(self, phrase, image_url):
        task = f"<grounding><phrase> {phrase} </phrase>"
        self.run(task, image_url)

    def referring_expression_generation(self, phrase, image_url):
        task = (
            "<grounding><phrase>"
            " It</phrase><object><patch_index_0044><patch_index_0863></object> is"
        )
        self.run(task, image_url)

    def grounded_vqa(self, question, image_url):
        task = f"<grounding> Question: {question} Answer:"
        self.run(task, image_url)

    def grounded_image_captioning(self, image_url):
        task = "<grounding> An image of"
        self.run(task, image_url)

    def grounded_image_captioning_detailed(self, image_url):
        task = "<grounding> Describe this image in detail"
        self.run(task, image_url)

    def draw_entity_boxes_on_image(
        image, entities, show=False, save_path=None
    ):
        """_summary_
        Args:
            image (_type_): image or image path
            collect_entity_location (_type_): _description_
        """
        if isinstance(image, Image.Image):
            image_h = image.height
            image_w = image.width
            image = np.array(image)[:, :, [2, 1, 0]]
        elif isinstance(image, str):
            if os.path.exists(image):
                pil_img = Image.open(image).convert("RGB")
                image = np.array(pil_img)[:, :, [2, 1, 0]]
                image_h = pil_img.height
                image_w = pil_img.width
            else:
                raise ValueError(f"invaild image path, {image}")
        elif isinstance(image, torch.Tensor):
            # pdb.set_trace()
            image_tensor = image.cpu()
            reverse_norm_mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073]
            )[:, None, None]
            reverse_norm_std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711]
            )[:, None, None]
            image_tensor = (
                image_tensor * reverse_norm_std + reverse_norm_mean
            )
            pil_img = T.ToPILImage()(image_tensor)
            image_h = pil_img.height
            image_w = pil_img.width
            image = np.array(pil_img)[:, :, [2, 1, 0]]
        else:
            raise ValueError(
                f"invaild image format, {type(image)} for {image}"
            )

        if len(entities) == 0:
            return image

        new_image = image.copy()
        previous_bboxes = []
        # size of text
        text_size = 1
        # thickness of text
        text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
        box_line = 3
        (c_width, text_height), _ = cv2.getTextSize(
            "F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line
        )
        base_height = int(text_height * 0.675)
        text_offset_original = text_height - base_height
        text_spaces = 3

        for entity_name, (start, end), bboxes in entities:
            for x1_norm, y1_norm, x2_norm, y2_norm in bboxes:
                orig_x1, orig_y1, orig_x2, orig_y2 = (
                    int(x1_norm * image_w),
                    int(y1_norm * image_h),
                    int(x2_norm * image_w),
                    int(y2_norm * image_h),
                )
                # draw bbox
                # random color
                color = tuple(
                    np.random.randint(0, 255, size=3).tolist()
                )
                new_image = cv2.rectangle(
                    new_image,
                    (orig_x1, orig_y1),
                    (orig_x2, orig_y2),
                    color,
                    box_line,
                )

                l_o, r_o = (
                    box_line // 2 + box_line % 2,
                    box_line // 2 + box_line % 2 + 1,
                )

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if (
                    y1
                    < text_height
                    + text_offset_original
                    + 2 * text_spaces
                ):
                    y1 = (
                        orig_y1
                        + r_o
                        + text_height
                        + text_offset_original
                        + 2 * text_spaces
                    )
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(
                    f"  {entity_name}",
                    cv2.FONT_HERSHEY_COMPLEX,
                    text_size,
                    text_line,
                )
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = (
                    x1,
                    y1
                    - (
                        text_height
                        + text_offset_original
                        + 2 * text_spaces
                    ),
                    x1 + text_width,
                    y1,
                )

                for prev_bbox in previous_bboxes:
                    while is_overlapping(
                        (
                            text_bg_x1,
                            text_bg_y1,
                            text_bg_x2,
                            text_bg_y2,
                        ),
                        prev_bbox,
                    ):
                        text_bg_y1 += (
                            text_height
                            + text_offset_original
                            + 2 * text_spaces
                        )
                        text_bg_y2 += (
                            text_height
                            + text_offset_original
                            + 2 * text_spaces
                        )
                        y1 += (
                            text_height
                            + text_offset_original
                            + 2 * text_spaces
                        )

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(
                                0,
                                image_h
                                - (
                                    text_height
                                    + text_offset_original
                                    + 2 * text_spaces
                                ),
                            )
                            text_bg_y2 = image_h
                            y1 = image_h
                            break

                alpha = 0.5
                for i in range(text_bg_y1, text_bg_y2):
                    for j in range(text_bg_x1, text_bg_x2):
                        if i < image_h and j < image_w:
                            if j < text_bg_x1 + 1.35 * c_width:
                                # original color
                                bg_color = color
                            else:
                                # white
                                bg_color = [255, 255, 255]
                            new_image[i, j] = (
                                alpha * new_image[i, j]
                                + (1 - alpha) * np.array(bg_color)
                            ).astype(np.uint8)

                cv2.putText(
                    new_image,
                    f"  {entity_name}",
                    (x1, y1 - text_offset_original - 1 * text_spaces),
                    cv2.FONT_HERSHEY_COMPLEX,
                    text_size,
                    (0, 0, 0),
                    text_line,
                    cv2.LINE_AA,
                )
                # previous_locations.append((x1, y1))
                previous_bboxes.append(
                    (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2)
                )

        pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
        if save_path:
            pil_image.save(save_path)
        if show:
            pil_image.show()

        return new_image

    def generate_boxees(self, task, image_url):
        image = self.get_image(image_url)
        processed_text, entities = self.process_task(task, image)
        self.draw_entity_boxes_on_image(image, entities, show=True)
