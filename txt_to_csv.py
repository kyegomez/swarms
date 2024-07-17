import os
import pandas as pd

# Specify the directory you want to use
directory = "emails"

# Create an empty DataFrame to store all data
all_data = pd.DataFrame(columns=["Email"])

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .txt file
    if filename.endswith(".txt"):
        # Construct the full file path
        filepath = os.path.join(directory, filename)

        # Read the file into a pandas DataFrame
        df = pd.read_csv(filepath, sep="\t", header=None)

        # Rename the column
        df.columns = ["Email"]

        # Append the data to the all_data DataFrame
        all_data = all_data._append(df, ignore_index=True)

# Construct the output file path
output_filepath = os.path.join(directory, "combined.csv")

# Write the DataFrame to a .csv file
all_data.to_csv(output_filepath, index=False)
