from swarms.models import LayoutLMDocumentQA

model = LayoutLMDocumentQA()

# Place an image of a financial document
out = model("What is the total amount?", "images/swarmfest.png")

print(out)
