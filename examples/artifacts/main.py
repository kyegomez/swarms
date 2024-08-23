from swarms import Artifact

# Example usage
artifact = Artifact(file_path="example.txt", file_type=".txt")
artifact.create("Initial content")
artifact.edit("First edit")
artifact.edit("Second edit")
artifact.save()

# Export to JSON
artifact.export_to_json("artifact.json")

# Import from JSON
imported_artifact = Artifact.import_from_json("artifact.json")

# # Get metrics
print(artifact.get_metrics())
