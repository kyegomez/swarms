import json
import csv

with open("profession_personas.progress.json", "r") as file:
    data = json.load(file)

# Extract the professions list from the JSON structure
professions = data["professions"]

with open("data_personas_progress.csv", "w", newline="") as file:
    writer = csv.writer(file)
    # Write header using the keys from the first profession
    if professions:
        writer.writerow(professions[0].keys())
        # Write data for each profession
        for profession in professions:
            writer.writerow(profession.values())
