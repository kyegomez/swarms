import json
import csv

with open("profession_personas.progress.json", "r") as file:
    data = json.load(file)

# Extract the professions list from the JSON structure
professions = data["professions"]

with open("data_personas_progress.csv", "w", newline="") as file:
    # Write header using the keys from the first profession
    if professions:
        writer = csv.DictWriter(
            file, fieldnames=professions[0].keys()
        )
        writer.writeheader()
        # Write data for each profession
        writer.writerows(professions)
