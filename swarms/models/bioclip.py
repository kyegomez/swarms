"""


BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
BiomedCLIP is a biomedical vision-language foundation model that is pretrained on PMC-15M,
a dataset of 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central, using contrastive learning. It uses PubMedBERT as the text encoder and Vision Transformer as the image encoder, with domain-specific adaptations. It can perform various vision-language processing (VLP) tasks such as cross-modal retrieval, image classification, and visual question answering. BiomedCLIP establishes new state of the art in a wide range of standard datasets, and substantially outperforms prior VLP approaches:



Citation
@misc{https://doi.org/10.48550/arXiv.2303.00915,
  doi = {10.48550/ARXIV.2303.00915},
  url = {https://arxiv.org/abs/2303.00915},
  author = {Zhang, Sheng and Xu, Yanbo and Usuyama, Naoto and Bagga, Jaspreet and Tinn, Robert and Preston, Sam and Rao, Rajesh and Wei, Mu and Valluri, Naveen and Wong, Cliff and Lungren, Matthew and Naumann, Tristan and Poon, Hoifung},
  title = {Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing},
  publisher = {arXiv},
  year = {2023},
}

Model Use
How to use
Please refer to this example notebook.

Intended Use
This model is intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the experimental results reported in the reference paper.

Primary Intended Use
The primary intended use is to support AI researchers building on top of this work. BiomedCLIP and its associated models should be helpful for exploring various biomedical VLP research questions, especially in the radiology domain.

Out-of-Scope Use
Any deployed use case of the model --- commercial or otherwise --- is currently out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are not intended for deployed use cases. Please refer to the associated paper for more details.

Data
This model builds upon PMC-15M dataset, which is a large-scale parallel image-text dataset for biomedical vision-language processing. It contains 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central. It covers a diverse range of biomedical image types, such as microscopy, radiography, histology, and more.

Limitations
This model was developed using English corpora, and thus can be considered English-only.

Further information
Please refer to the corresponding paper, "Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing" for additional details on the model training and evaluation.
"""

import open_clip
import torch
from PIL import Image
import matplotlib.pyplot as plt


class BioClip:
    """
    BioClip

    Args:
        model_path (str): path to the model

    Attributes:
        model_path (str): path to the model
        model (torch.nn.Module): the model
        preprocess_train (torchvision.transforms.Compose): the preprocessing pipeline for training
        preprocess_val (torchvision.transforms.Compose): the preprocessing pipeline for validation
        tokenizer (open_clip.Tokenizer): the tokenizer
        device (torch.device): the device to run the model on

    Methods:
        __call__(self, img_path: str, labels: list, template: str = 'this is a photo of ', context_length: int = 256):
            returns a dictionary of labels and their probabilities
        plot_image_with_metadata(img_path: str, metadata: dict): plots the image with the metadata

    Usage:
        clip = BioClip('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        labels = [
            'adenocarcinoma histopathology',
            'brain MRI',
            'covid line chart',
            'squamous cell carcinoma histopathology',
            'immunohistochemistry histopathology',
            'bone X-ray',
            'chest X-ray',
            'pie chart',
            'hematoxylin and eosin histopathology'
        ]

        result = clip("your_image_path.jpg", labels)
        metadata = {'filename': "your_image_path.jpg".split('/')[-1], 'top_probs': result}
        clip.plot_image_with_metadata("your_image_path.jpg", metadata)


    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        (
            self.model,
            self.preprocess_train,
            self.preprocess_val,
        ) = open_clip.create_model_and_transforms(model_path)
        self.tokenizer = open_clip.get_tokenizer(model_path)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        img_path: str,
        labels: list,
        template: str = "this is a photo of ",
        context_length: int = 256,
    ):
        image = torch.stack(
            [self.preprocess_val(Image.open(img_path))]
        ).to(self.device)
        texts = self.tokenizer(
            [template + l for l in labels],
            context_length=context_length,
        ).to(self.device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(
                image, texts
            )
            logits = (
                (logit_scale * image_features @ text_features.t())
                .detach()
                .softmax(dim=-1)
            )
            sorted_indices = torch.argsort(
                logits, dim=-1, descending=True
            )
            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()

        results = {}
        for idx in sorted_indices[0]:
            label = labels[idx]
            prob = logits[0][idx]
            results[label] = prob
        return results

    @staticmethod
    def plot_image_with_metadata(img_path: str, metadata: dict):
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.axis("off")
        title = (
            metadata["filename"]
            + "\n"
            + "\n".join(
                [
                    f"{k}: {v*100:.1f}"
                    for k, v in metadata["top_probs"].items()
                ]
            )
        )
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plt.show()


# Usage
# clip = BioClip('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# labels = [
#     'adenocarcinoma histopathology',
#     'brain MRI',
#     'covid line chart',
#     'squamous cell carcinoma histopathology',
#     'immunohistochemistry histopathology',
#     'bone X-ray',
#     'chest X-ray',
#     'pie chart',
#     'hematoxylin and eosin histopathology'
# ]

# result = clip("your_image_path.jpg", labels)
# metadata = {'filename': "your_image_path.jpg".split('/')[-1], 'top_probs': result}
# clip.plot_image_with_metadata("your_image_path.jpg", metadata)
