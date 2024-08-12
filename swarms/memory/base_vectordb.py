from abc import ABC
from swarms.utils.loguru_logger import logger


class BaseVectorDatabase(ABC):
    """
    Abstract base class for a database.

    This class defines the interface for interacting with a database.
    Subclasses must implement the abstract methods to provide the
    specific implementation details for connecting to a database,
    executing queries, and performing CRUD operations.

    """

    def connect(self):
        """
        Connect to the database.

        This method establishes a connection to the database.

        """

    def close(self):
        """
        Close the database connection.

        This method closes the connection to the database.

        """

    def query(self, query: str):
        """
        Execute a database query.

        This method executes the given query on the database.

        Parameters:
            query (str): The query to be executed.

        """

    def fetch_all(self):
        """
        Fetch all rows from the result set.

        This method retrieves all rows from the result set of a query.

        Returns:
            list: A list of dictionaries representing the rows.

        """

    def fetch_one(self):
        """
        Fetch one row from the result set.

        This method retrieves one row from the result set of a query.

        Returns:
            dict: A dictionary representing the row.

        """

    def add(self, doc: str):
        """
        Add a new record to the database.

        This method adds a new record to the specified table in the database.

        Parameters:
            table (str): The name of the table.
            data (dict): A dictionary representing the data to be added.

        """

    def get(self, query: str):
        """
        Get a record from the database.

        This method retrieves a record from the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be retrieved.

        Returns:
            dict: A dictionary representing the retrieved record.

        """

    def update(self, doc):
        """
        Update a record in the database.

        This method updates a record in the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be updated.
            data (dict): A dictionary representing the updated data.

        """

    def delete(self, message):
        """
        Delete a record from the database.

        This method deletes a record from the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be deleted.

        """

    def print_all(self):
        """
        Print all records in the database.

        This method prints all records in the specified table in the database.

        """
        pass

    def log_query(self, query: str = None):
        """
        Log the query.

        This method logs the query that was executed on the database.

        Parameters:
            query (str): The query that was executed.

        """
        logger.info(f"Query: {query}")

    def log_retrieved_data(self, data: list = None):
        """
        Log the retrieved data.

        This method logs the data that was retrieved from the database.

        Parameters:
            data (dict): The data that was retrieved.

        """
        for d in data:
            logger.info(f"Retrieved Data: {d}")
