# Text embeddings, image embeddings, and multimodal embeddings
# Add text and image embeddings into postgresl database

from swarms.models.jina_embeds import JinaEmbeddings
from swarms.models.gigabind import Gigabind

# Model
model = JinaEmbeddings(
    max_length=8192,
    device="cuda",
    quantize=True,
    huggingface_api_key="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",
)


# Encode text

embeddings = model("Encode this super long document text")


# Embed images or text
model = Gigabind()

multi_modal_embeddings = model(text=[text], imgs=[img1, img2, img3])
