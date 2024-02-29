import pandas as pd
from swarms import dataframe_to_text

# # Example usage:
df = pd.DataFrame(
    {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
    }
)

print(dataframe_to_text(df))
