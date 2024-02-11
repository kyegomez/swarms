from abc import ABC, abstractmethod


class AbstractVectorDatabase(ABC):
    """
    Abstract base class for a database.

    This class defines the interface for interacting with a database.
    Subclasses must implement the abstract methods to provide the
    specific implementation details for connecting to a database,
    executing queries, and performing CRUD operations.

    """

    @abstractmethod
    def connect(self):
        """
        Connect to the database.

        This method establishes a connection to the database.

        """

        pass

    @abstractmethod
    def close(self):
        """
        Close the database connection.

        This method closes the connection to the database.

        """

        pass

    @abstractmethod
    def query(self, query: str):
        """
        Execute a database query.

        This method executes the given query on the database.

        Parameters:
            query (str): The query to be executed.

        """

        pass

    @abstractmethod
    def fetch_all(self):
        """
        Fetch all rows from the result set.

        This method retrieves all rows from the result set of a query.

        Returns:
            list: A list of dictionaries representing the rows.

        """

        pass

    @abstractmethod
    def fetch_one(self):
        """
        Fetch one row from the result set.

        This method retrieves one row from the result set of a query.

        Returns:
            dict: A dictionary representing the row.

        """

        pass

    @abstractmethod
    def add(self, doc: str):
        """
        Add a new record to the database.

        This method adds a new record to the specified table in the database.

        Parameters:
            table (str): The name of the table.
            data (dict): A dictionary representing the data to be added.

        """

        pass

    @abstractmethod
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

        pass

    @abstractmethod
    def update(self, doc):
        """
        Update a record in the database.

        This method updates a record in the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be updated.
            data (dict): A dictionary representing the updated data.

        """

        pass

    @abstractmethod
    def delete(self, message):
        """
        Delete a record from the database.

        This method deletes a record from the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be deleted.

        """

        pass
