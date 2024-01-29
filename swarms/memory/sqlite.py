from typing import List, Tuple, Any, Optional
from swarms.memory.base_vectordb import AbstractVectorDatabase

try:
    import sqlite3
except ImportError:
    raise ImportError(
        "Please install sqlite3 to use the SQLiteDB class."
    )


class SQLiteDB(AbstractVectorDatabase):
    """
    A reusable class for SQLite database operations with methods for adding,
    deleting, updating, and querying data.

    Attributes:
        db_path (str): The file path to the SQLite database.
    """

    def __init__(self, db_path: str):
        """
        Initializes the SQLiteDB class with the given database path.

        Args:
            db_path (str): The file path to the SQLite database.
        """
        self.db_path = db_path

    def execute_query(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> List[Tuple]:
        """
        Executes a SQL query and returns fetched results.

        Args:
            query (str): The SQL query to execute.
            params (Tuple[Any, ...], optional): The parameters to substitute into the query.

        Returns:
            List[Tuple]: The results fetched from the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                return cursor.fetchall()
        except Exception as error:
            print(f"Error executing query: {error}")
            raise error

    def add(self, query: str, params: Tuple[Any, ...]) -> None:
        """
        Adds a new entry to the database.

        Args:
            query (str): The SQL query for insertion.
            params (Tuple[Any, ...]): The parameters to substitute into the query.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
        except Exception as error:
            print(f"Error adding new entry: {error}")
            raise error

    def delete(self, query: str, params: Tuple[Any, ...]) -> None:
        """
        Deletes an entry from the database.

        Args:
            query (str): The SQL query for deletion.
            params (Tuple[Any, ...]): The parameters to substitute into the query.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
        except Exception as error:
            print(f"Error deleting entry: {error}")
            raise error

    def update(self, query: str, params: Tuple[Any, ...]) -> None:
        """
        Updates an entry in the database.

        Args:
            query (str): The SQL query for updating.
            params (Tuple[Any, ...]): The parameters to substitute into the query.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
        except Exception as error:
            print(f"Error updating entry: {error}")
            raise error

    def query(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> List[Tuple]:
        """
        Fetches data from the database based on a query.

        Args:
            query (str): The SQL query to execute.
            params (Tuple[Any, ...], optional): The parameters to substitute into the query.

        Returns:
            List[Tuple]: The results fetched from the database.
        """
        try:
            return self.execute_query(query, params)
        except Exception as error:
            print(f"Error querying database: {error}")
            raise error
