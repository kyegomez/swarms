import pandas as pd
import hashlib


def dataframe_to_hash(dataframe: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(dataframe, index=True).values
    ).hexdigest()


def str_to_hash(text: str, hash_algorithm: str = "sha256") -> str:
    m = hashlib.new(hash_algorithm)

    m.update(text.encode())

    return m.hexdigest()
