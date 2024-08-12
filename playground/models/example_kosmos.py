from swarms import Kosmos

# Initialize the model
model = Kosmos()

# Generate
out = model.run("Analyze the reciepts in this image", "docs.jpg")

# Print the output
print(out)
