from swarms.models.kosmos_two import Kosmos

# Initialize Kosmos
kosmos = Kosmos()

# Perform multimodal grounding
out = kosmos.multimodal_grounding(
    "Find the red apple in the image.", "images/swarms.jpeg"
)

print(out)
