from abc import ABC, abstractmethod


class AbstractDatabase(ABC):
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

    @abstractmethod
    def close(self):
        """
        Close the database connection.

        This method closes the connection to the database.

        """

    @abstractmethod
    def execute_query(self, query):
        """
        Execute a database query.

        This method executes the given query on the database.

        Parameters:
            query (str): The query to be executed.

        """

    @abstractmethod
    def fetch_all(self):
        """
        Fetch all rows from the result set.

        This method retrieves all rows from the result set of a query.

        Returns:
            list: A list of dictionaries representing the rows.

        """

    @abstractmethod
    def fetch_one(self):
        """
        Fetch one row from the result set.

        This method retrieves one row from the result set of a query.

        Returns:
            dict: A dictionary representing the row.

        """

    @abstractmethod
    def add(self, table, data):
        """
        Add a new record to the database.

        This method adds a new record to the specified table in the database.

        Parameters:
            table (str): The name of the table.
            data (dict): A dictionary representing the data to be added.

        """

    @abstractmethod
    def query(self, table, condition):
        """
        Query the database.

        This method queries the specified table in the database based on the given condition.

        Parameters:
            table (str): The name of the table.
            condition (str): The condition to be applied in the query.

        Returns:
            list: A list of dictionaries representing the query results.

        """

    @abstractmethod
    def get(self, table, id):
        """
        Get a record from the database.

        This method retrieves a record from the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be retrieved.

        Returns:
            dict: A dictionary representing the retrieved record.

        """

    @abstractmethod
    def update(self, table, id, data):
        """
        Update a record in the database.

        This method updates a record in the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be updated.
            data (dict): A dictionary representing the updated data.

        """

    @abstractmethod
    def delete(self, table, id):
        """
        Delete a record from the database.

        This method deletes a record from the specified table in the database based on the given ID.

        Parameters:
            table (str): The name of the table.
            id (int): The ID of the record to be deleted.

        """
