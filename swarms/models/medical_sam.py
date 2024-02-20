import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
from skimage import transform
from torch import Tensor


def sam_model_registry():
    pass


@dataclass
class MedicalSAM:
    """
    MedicalSAM class for performing semantic segmentation on medical images using the SAM model.

    Attributes:
        model_path (str): The file path to the model weights.
        device (str): The device to run the model on (default is "cuda:0").
        model_weights_url (str): The URL to download the model weights from.

    Methods:
        __post_init__(): Initializes the MedicalSAM object.
        download_model_weights(model_path: str): Downloads the model weights from the specified URL and saves them to the given file path.
        preprocess(img): Preprocesses the input image.
        run(img, box): Runs the semantic segmentation on the input image within the specified bounding box.

    """

    model_path: str
    device: str = "cuda:0"
    model_weights_url: str = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    def __post_init__(self):
        if not os.path.exists(self.model_path):
            self.download_model_weights(self.model_path)

        self.model = sam_model_registry["vit_b"](
            checkpoint=self.model_path
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def download_model_weights(self, model_path: str):
        """
        Downloads the model weights from the specified URL and saves them to the given file path.

        Args:
            model_path (str): The file path where the model weights will be saved.

        Raises:
            Exception: If the model weights fail to download.
        """
        response = requests.get(self.model_weights_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception("Failed to download model weights.")

    def preprocess(self, img: np.ndarray) -> Tuple[Tensor, int, int]:
        """
        Preprocesses the input image.

        Args:
            img: The input image.

        Returns:
            img_tensor: The preprocessed image tensor.
            H: The original height of the image.
            W: The original width of the image.
        """
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        H, W, _ = img.shape
        img = transform.resize(
            img,
            (1024, 1024),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
        img = img - img.min() / np.clip(
            img.max() - img.min(), a_min=1e-8, a_max=None
        )
        img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0)
        return img, H, W

    @torch.no_grad()
    def run(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Runs the semantic segmentation on the input image within the specified bounding box.

        Args:
            img: The input image.
            box: The bounding box coordinates (x1, y1, x2, y2).

        Returns:
            medsam_seg: The segmented image.
        """
        img_tensor, H, W = self.preprocess(img)
        img_tensor = img_tensor.to(self.device)
        box_1024 = box / np.array([W, H, W, H]) * 1024
        img = self.model.image_encoder(img_tensor)

        box_torch = torch.as_tensor(
            box_1024, dtype=torch.float, device=img_tensor.device
        )

        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=img,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

        return medsam_seg
