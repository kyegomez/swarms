import pandas as pd


# CSV to dataframe
def csv_to_dataframe(file_path):
    """
    Read a CSV file and return a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df


# Dataframe to strings
def dataframe_to_strings(df):
    """
    Converts a pandas DataFrame to a list of string representations of each row.

    Args:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        list: A list of string representations of each row in the DataFrame.
    """
    row_strings = []
    for index, row in df.iterrows():
        row_string = row.to_string()
        row_strings.append(row_string)
    return row_strings
