from swarms.models.bioclip import BioClip

clip = BioClip(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

labels = [
    "adenocarcinoma histopathology",
    "brain MRI",
    "covid line chart",
    "squamous cell carcinoma histopathology",
    "immunohistochemistry histopathology",
    "bone X-ray",
    "chest X-ray",
    "pie chart",
    "hematoxylin and eosin histopathology",
]

result = clip("swarms.jpeg", labels)
metadata = {
    "filename": "images/.jpg".split("/")[-1],
    "top_probs": result,
}
clip.plot_image_with_metadata("swarms.jpeg", metadata)
