import pandas as pd


def dataframe_to_text(
    df: pd.DataFrame,
    parsing_func: callable = None,
) -> str:
    """
    Convert a pandas DataFrame to a string representation.

    Args:
        df (pd.DataFrame): The pandas DataFrame to convert.
        parsing_func (callable, optional): A function to parse the resulting text. Defaults to None.

    Returns:
        str: The string representation of the DataFrame.

    Example:
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9],
    ... })
    >>> print(dataframe_to_text(df))

    """
    # Get a string representation of the dataframe
    df_str = df.to_string()

    # Get a string representation of the column names
    info_str = df.info()

    # Combine the dataframe string and the info string
    text = f"DataFrame:\n{df_str}\n\nInfo:\n{info_str}"

    if parsing_func:
        text = parsing_func(text)

    return text


# # # Example usage:
# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4, 5, 6],
#     'C': [7, 8, 9],
# })

# print(dataframe_to_text(df))
