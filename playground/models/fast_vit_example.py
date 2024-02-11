from swarms.models.fastvit import FastViT

fastvit = FastViT()

result = fastvit(img="images/swarms.jpeg", confidence_threshold=0.5)
