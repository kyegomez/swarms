# Smart Database Powered by Hierarchical Multi-Agent Workflow

This module implements a fully autonomous database management system using a hierarchical
multi-agent architecture. The system includes specialized agents for different database
operations coordinated by a Database Director agent.

## Features

| Feature                              | Description                                                                                   |
|---------------------------------------|-----------------------------------------------------------------------------------------------|
| Autonomous Database Management        | Complete database lifecycle management, including setup and ongoing management of databases.   |
| Intelligent Task Distribution         | Automatic assignment of tasks to appropriate specialist agents.                                |
| Table Creation with Schema Validation | Ensures tables are created with correct structure, schema enforcement, and data integrity.     |
| Data Insertion and Updates            | Handles adding new data and updating existing records efficiently, supporting JSON input.      |
| Complex Query Execution               | Executes advanced and optimized queries for data retrieval and analysis.                       |
| Schema Modifications                  | Supports altering table structures and database schemas as needed.                             |
| Hierarchical Agent Coordination       | Utilizes a multi-agent system for orchestrated, intelligent task execution.                    |
| Security                              | Built-in SQL injection prevention and query validation for data protection.                    |
| Performance Optimization              | Query optimization and efficient data operations for high performance.                         |
| Comprehensive Error Handling          | Robust error management and reporting throughout all operations.                               |
| Multi-format Data Support             | Flexible query parameters and support for JSON-based data insertion.                           |

## Architecture

### Multi-Agent Architecture

```
Database Director (Coordinator)
├── Database Creator (Creates databases)
├── Table Manager (Manages table schemas)
├── Data Operations (Handles data insertion/updates)
└── Query Specialist (Executes queries and retrieval)
```

### Agent Specializations

| Agent             | Description                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------|
| **Database Director**  | Orchestrates all database operations and coordinates specialist agents                        |
| **Database Creator**   | Specializes in creating and initializing databases                                            |
| **Table Manager**      | Expert in table creation, schema design, and structure management                             |
| **Data Operations**    | Handles data insertion, updates, and manipulation                                             |
| **Query Specialist**   | Manages database queries, data retrieval, and optimization                                    |


## Agent Tools

| Function | Description |
|----------|-------------|
| **`create_database(database_name, database_path)`** | Creates new SQLite databases |
| **`create_table(database_path, table_name, schema)`** | Creates tables with specified schemas |
| **`insert_data(database_path, table_name, data)`** | Inserts data into tables |
| **`query_database(database_path, query, params)`** | Executes SELECT queries |
| **`update_table_data(database_path, table_name, update_data, where_clause)`** | Updates existing data |
| **`get_database_schema(database_path)`** | Retrieves comprehensive schema information |

## Install

```bash
pip install -U swarms sqlite3 loguru
```

## ENV

```
WORKSPACE_DIR="agent_workspace"
ANTHROPIC_API_KEY=""
OPENAI_API_KEY=""
```

## Code 

- Make a file called `smart_database_swarm.py`

```python
import sqlite3
import json
from pathlib import Path
from loguru import logger

from swarms import Agent, HierarchicalSwarm


# =============================================================================
# DATABASE TOOLS - Core Functions for Database Operations
# =============================================================================


def create_database(
    database_name: str, database_path: str = "./databases"
) -> str:
    """
    Create a new SQLite database file.

    Args:
        database_name (str): Name of the database to create (without .db extension)
        database_path (str, optional): Directory path where database will be created.
                                     Defaults to "./databases".

    Returns:
        str: JSON string containing operation result and database information

    Raises:
        OSError: If unable to create database directory or file
        sqlite3.Error: If database connection fails

    Example:
        >>> result = create_database("company_db", "/data/databases")
        >>> print(result)
        {"status": "success", "database": "company_db.db", "path": "/data/databases/company_db.db"}
    """
    try:
        # Validate input parameters
        if not database_name or not database_name.strip():
            raise ValueError("Database name cannot be empty")

        # Clean database name
        db_name = database_name.strip().replace(" ", "_")
        if not db_name.endswith(".db"):
            db_name += ".db"

        # Create database directory if it doesn't exist
        db_path = Path(database_path)
        db_path.mkdir(parents=True, exist_ok=True)

        # Full database file path
        full_db_path = db_path / db_name

        # Create database connection (creates file if doesn't exist)
        conn = sqlite3.connect(str(full_db_path))

        # Create a metadata table to track database info
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _database_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Insert database metadata
        conn.execute(
            "INSERT OR REPLACE INTO _database_metadata (key, value) VALUES (?, ?)",
            ("database_name", database_name),
        )

        conn.commit()
        conn.close()

        result = {
            "status": "success",
            "message": f"Database '{database_name}' created successfully",
            "database": db_name,
            "path": str(full_db_path),
            "size_bytes": full_db_path.stat().st_size,
        }

        logger.info(f"Database created: {db_name}")
        return json.dumps(result, indent=2)

    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})
    except sqlite3.Error as e:
        return json.dumps(
            {"status": "error", "error": f"Database error: {str(e)}"}
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }
        )


def create_table(
    database_path: str, table_name: str, schema: str
) -> str:
    """
    Create a new table in the specified database with the given schema.

    Args:
        database_path (str): Full path to the database file
        table_name (str): Name of the table to create
        schema (str): SQL schema definition for the table columns
                     Format: "column1 TYPE constraints, column2 TYPE constraints, ..."
                     Example: "id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER"

    Returns:
        str: JSON string containing operation result and table information

    Raises:
        sqlite3.Error: If table creation fails
        FileNotFoundError: If database file doesn't exist

    Example:
        >>> schema = "id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE"
        >>> result = create_table("/data/company.db", "employees", schema)
        >>> print(result)
        {"status": "success", "table": "employees", "columns": 3}
    """
    try:
        # Validate inputs
        if not all([database_path, table_name, schema]):
            raise ValueError(
                "Database path, table name, and schema are required"
            )

        # Check if database exists
        if not Path(database_path).exists():
            raise FileNotFoundError(
                f"Database file not found: {database_path}"
            )

        # Clean table name
        clean_table_name = table_name.strip().replace(" ", "_")

        # Connect to database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Check if table already exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (clean_table_name,),
        )

        if cursor.fetchone():
            conn.close()
            return json.dumps(
                {
                    "status": "warning",
                    "message": f"Table '{clean_table_name}' already exists",
                    "table": clean_table_name,
                }
            )

        # Create table with provided schema
        create_sql = f"CREATE TABLE {clean_table_name} ({schema})"
        cursor.execute(create_sql)

        # Get table info
        cursor.execute(f"PRAGMA table_info({clean_table_name})")
        columns = cursor.fetchall()

        # Update metadata
        cursor.execute(
            """
            INSERT OR REPLACE INTO _database_metadata (key, value) 
            VALUES (?, ?)
        """,
            (f"table_{clean_table_name}_created", "true"),
        )

        conn.commit()
        conn.close()

        result = {
            "status": "success",
            "message": f"Table '{clean_table_name}' created successfully",
            "table": clean_table_name,
            "columns": len(columns),
            "schema": [
                {
                    "name": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                }
                for col in columns
            ],
        }

        return json.dumps(result, indent=2)

    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})
    except FileNotFoundError as e:
        return json.dumps({"status": "error", "error": str(e)})
    except sqlite3.Error as e:
        return json.dumps(
            {"status": "error", "error": f"SQL error: {str(e)}"}
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }
        )


def insert_data(
    database_path: str, table_name: str, data: str
) -> str:
    """
    Insert data into a specified table.

    Args:
        database_path (str): Full path to the database file
        table_name (str): Name of the target table
        data (str): JSON string containing data to insert
                   Format: {"columns": ["col1", "col2"], "values": [[val1, val2], ...]}
                   Or: [{"col1": val1, "col2": val2}, ...]

    Returns:
        str: JSON string containing operation result and insertion statistics

    Example:
        >>> data = '{"columns": ["name", "age"], "values": [["John", 30], ["Jane", 25]]}'
        >>> result = insert_data("/data/company.db", "employees", data)
        >>> print(result)
        {"status": "success", "table": "employees", "rows_inserted": 2}
    """
    try:
        # Validate inputs
        if not all([database_path, table_name, data]):
            raise ValueError(
                "Database path, table name, and data are required"
            )

        # Check if database exists
        if not Path(database_path).exists():
            raise FileNotFoundError(
                f"Database file not found: {database_path}"
            )

        # Parse data
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for data")

        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )

        if not cursor.fetchone():
            conn.close()
            raise ValueError(f"Table '{table_name}' does not exist")

        rows_inserted = 0

        # Handle different data formats
        if isinstance(parsed_data, list) and all(
            isinstance(item, dict) for item in parsed_data
        ):
            # Format: [{"col1": val1, "col2": val2}, ...]
            for row in parsed_data:
                columns = list(row.keys())
                values = list(row.values())
                placeholders = ", ".join(["?" for _ in values])
                columns_str = ", ".join(columns)

                insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                cursor.execute(insert_sql, values)
                rows_inserted += 1

        elif (
            isinstance(parsed_data, dict)
            and "columns" in parsed_data
            and "values" in parsed_data
        ):
            # Format: {"columns": ["col1", "col2"], "values": [[val1, val2], ...]}
            columns = parsed_data["columns"]
            values_list = parsed_data["values"]

            placeholders = ", ".join(["?" for _ in columns])
            columns_str = ", ".join(columns)

            insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            for values in values_list:
                cursor.execute(insert_sql, values)
                rows_inserted += 1
        else:
            raise ValueError(
                "Invalid data format. Expected list of dicts or dict with columns/values"
            )

        conn.commit()
        conn.close()

        result = {
            "status": "success",
            "message": f"Data inserted successfully into '{table_name}'",
            "table": table_name,
            "rows_inserted": rows_inserted,
        }

        return json.dumps(result, indent=2)

    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"status": "error", "error": str(e)})
    except sqlite3.Error as e:
        return json.dumps(
            {"status": "error", "error": f"SQL error: {str(e)}"}
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }
        )


def query_database(
    database_path: str, query: str, params: str = "[]"
) -> str:
    """
    Execute a SELECT query on the database and return results.

    Args:
        database_path (str): Full path to the database file
        query (str): SQL SELECT query to execute
        params (str, optional): JSON string of query parameters for prepared statements.
                               Defaults to "[]".

    Returns:
        str: JSON string containing query results and metadata

    Example:
        >>> query = "SELECT * FROM employees WHERE age > ?"
        >>> params = "[25]"
        >>> result = query_database("/data/company.db", query, params)
        >>> print(result)
        {"status": "success", "results": [...], "row_count": 5}
    """
    try:
        # Validate inputs
        if not all([database_path, query]):
            raise ValueError("Database path and query are required")

        # Check if database exists
        if not Path(database_path).exists():
            raise FileNotFoundError(
                f"Database file not found: {database_path}"
            )

        # Validate query is SELECT only (security)
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        # Parse parameters
        try:
            query_params = json.loads(params)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for parameters")

        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()

        # Execute query
        if query_params:
            cursor.execute(query, query_params)
        else:
            cursor.execute(query)

        # Fetch results
        rows = cursor.fetchall()

        # Convert to list of dictionaries
        results = [dict(row) for row in rows]

        # Get column names
        column_names = (
            [description[0] for description in cursor.description]
            if cursor.description
            else []
        )

        conn.close()

        result = {
            "status": "success",
            "message": "Query executed successfully",
            "results": results,
            "row_count": len(results),
            "columns": column_names,
        }

        return json.dumps(result, indent=2)

    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"status": "error", "error": str(e)})
    except sqlite3.Error as e:
        return json.dumps(
            {"status": "error", "error": f"SQL error: {str(e)}"}
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }
        )


def update_table_data(
    database_path: str,
    table_name: str,
    update_data: str,
    where_clause: str = "",
) -> str:
    """
    Update existing data in a table.

    Args:
        database_path (str): Full path to the database file
        table_name (str): Name of the table to update
        update_data (str): JSON string with column-value pairs to update
                          Format: {"column1": "new_value1", "column2": "new_value2"}
        where_clause (str, optional): WHERE condition for the update (without WHERE keyword).
                                    Example: "id = 1 AND status = 'active'"

    Returns:
        str: JSON string containing operation result and update statistics

    Example:
        >>> update_data = '{"salary": 50000, "department": "Engineering"}'
        >>> where_clause = "id = 1"
        >>> result = update_table_data("/data/company.db", "employees", update_data, where_clause)
        >>> print(result)
        {"status": "success", "table": "employees", "rows_updated": 1}
    """
    try:
        # Validate inputs
        if not all([database_path, table_name, update_data]):
            raise ValueError(
                "Database path, table name, and update data are required"
            )

        # Check if database exists
        if not Path(database_path).exists():
            raise FileNotFoundError(
                f"Database file not found: {database_path}"
            )

        # Parse update data
        try:
            parsed_updates = json.loads(update_data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for update data")

        if not isinstance(parsed_updates, dict):
            raise ValueError("Update data must be a dictionary")

        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )

        if not cursor.fetchone():
            conn.close()
            raise ValueError(f"Table '{table_name}' does not exist")

        # Build UPDATE query
        set_clauses = []
        values = []

        for column, value in parsed_updates.items():
            set_clauses.append(f"{column} = ?")
            values.append(value)

        set_clause = ", ".join(set_clauses)

        if where_clause:
            update_sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        else:
            update_sql = f"UPDATE {table_name} SET {set_clause}"

        # Execute update
        cursor.execute(update_sql, values)
        rows_updated = cursor.rowcount

        conn.commit()
        conn.close()

        result = {
            "status": "success",
            "message": f"Table '{table_name}' updated successfully",
            "table": table_name,
            "rows_updated": rows_updated,
            "updated_columns": list(parsed_updates.keys()),
        }

        return json.dumps(result, indent=2)

    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"status": "error", "error": str(e)})
    except sqlite3.Error as e:
        return json.dumps(
            {"status": "error", "error": f"SQL error: {str(e)}"}
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }
        )


def get_database_schema(database_path: str) -> str:
    """
    Get comprehensive schema information for all tables in the database.

    Args:
        database_path (str): Full path to the database file

    Returns:
        str: JSON string containing complete database schema information

    Example:
        >>> result = get_database_schema("/data/company.db")
        >>> print(result)
        {"status": "success", "database": "company.db", "tables": {...}}
    """
    try:
        if not database_path:
            raise ValueError("Database path is required")

        if not Path(database_path).exists():
            raise FileNotFoundError(
                f"Database file not found: {database_path}"
            )

        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '_%'"
        )
        tables = cursor.fetchall()

        schema_info = {
            "database": Path(database_path).name,
            "table_count": len(tables),
            "tables": {},
        }

        for table in tables:
            table_name = table[0]

            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            schema_info["tables"][table_name] = {
                "columns": [
                    {
                        "name": col[1],
                        "type": col[2],
                        "nullable": not col[3],
                        "default": col[4],
                        "primary_key": bool(col[5]),
                    }
                    for col in columns
                ],
                "column_count": len(columns),
                "row_count": row_count,
            }

        conn.close()

        result = {
            "status": "success",
            "message": "Database schema retrieved successfully",
            "schema": schema_info,
        }

        return json.dumps(result, indent=2)

    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"status": "error", "error": str(e)})
    except sqlite3.Error as e:
        return json.dumps(
            {"status": "error", "error": f"SQL error: {str(e)}"}
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }
        )


# =============================================================================
# DATABASE CREATION SPECIALIST AGENT
# =============================================================================
database_creator_agent = Agent(
    agent_name="Database-Creator",
    agent_description="Specialist agent responsible for creating and initializing databases with proper structure and metadata",
    system_prompt="""You are the Database Creator, a specialist agent responsible for database creation and initialization. Your expertise includes:

    DATABASE CREATION & SETUP:
    - Creating new SQLite databases with proper structure
    - Setting up database metadata and tracking systems
    - Initializing database directories and file organization
    - Ensuring database accessibility and permissions
    - Creating database backup and recovery procedures

    DATABASE ARCHITECTURE:
    - Designing optimal database structures for different use cases
    - Planning database organization and naming conventions
    - Setting up database configuration and optimization settings
    - Implementing database security and access controls
    - Creating database documentation and specifications

    Your responsibilities:
    - Create new databases when requested
    - Set up proper database structure and metadata
    - Ensure database is properly initialized and accessible
    - Provide database creation status and information
    - Handle database creation errors and provide solutions

    You work with precise technical specifications and always ensure databases are created correctly and efficiently.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.3,
    dynamic_temperature_enabled=True,
    tools=[create_database, get_database_schema],
)

# =============================================================================
# TABLE MANAGEMENT SPECIALIST AGENT
# =============================================================================
table_manager_agent = Agent(
    agent_name="Table-Manager",
    agent_description="Specialist agent for table creation, schema design, and table structure management",
    system_prompt="""You are the Table Manager, a specialist agent responsible for table creation, schema design, and table structure management. Your expertise includes:

    TABLE CREATION & DESIGN:
    - Creating tables with optimal schema design
    - Defining appropriate data types and constraints
    - Setting up primary keys, foreign keys, and indexes
    - Designing normalized table structures
    - Creating tables that support efficient queries and operations

    SCHEMA MANAGEMENT:
    - Analyzing schema requirements and designing optimal structures
    - Validating schema definitions and data types
    - Ensuring schema consistency and integrity
    - Managing schema modifications and updates
    - Optimizing table structures for performance

    DATA INTEGRITY:
    - Implementing proper constraints and validation rules
    - Setting up referential integrity between tables
    - Ensuring data consistency across table operations
    - Managing table relationships and dependencies
    - Creating tables that support data quality requirements

    Your responsibilities:
    - Create tables with proper schema definitions
    - Validate table structures and constraints
    - Ensure optimal table design for performance
    - Handle table creation errors and provide solutions
    - Provide detailed table information and metadata

    You work with precision and always ensure tables are created with optimal structure and performance characteristics.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.3,
    dynamic_temperature_enabled=True,
    tools=[create_table, get_database_schema],
)

# =============================================================================
# DATA OPERATIONS SPECIALIST AGENT
# =============================================================================
data_operations_agent = Agent(
    agent_name="Data-Operations",
    agent_description="Specialist agent for data insertion, updates, and data manipulation operations",
    system_prompt="""You are the Data Operations specialist, responsible for all data manipulation operations including insertion, updates, and data management. Your expertise includes:

    DATA INSERTION:
    - Inserting data with proper validation and formatting
    - Handling bulk data insertions efficiently
    - Managing data type conversions and formatting
    - Ensuring data integrity during insertion operations
    - Validating data before insertion to prevent errors

    DATA UPDATES:
    - Updating existing data with precision and safety
    - Creating targeted update operations with proper WHERE clauses
    - Managing bulk updates and data modifications
    - Ensuring data consistency during update operations
    - Validating update operations to prevent data corruption

    DATA VALIDATION:
    - Validating data formats and types before operations
    - Ensuring data meets schema requirements and constraints
    - Checking for data consistency and integrity
    - Managing data transformation and cleaning operations
    - Providing detailed feedback on data operation results

    ERROR HANDLING:
    - Managing data operation errors gracefully
    - Providing clear error messages and solutions
    - Ensuring data operations are atomic and safe
    - Rolling back operations when necessary
    - Maintaining data integrity throughout all operations

    Your responsibilities:
    - Execute data insertion operations safely and efficiently
    - Perform data updates with proper validation
    - Ensure data integrity throughout all operations
    - Handle data operation errors and provide solutions
    - Provide detailed operation results and statistics

    You work with extreme precision and always prioritize data integrity and safety in all operations.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.3,
    dynamic_temperature_enabled=True,
    tools=[insert_data, update_table_data],
)

# =============================================================================
# QUERY SPECIALIST AGENT
# =============================================================================
query_specialist_agent = Agent(
    agent_name="Query-Specialist",
    agent_description="Expert agent for database querying, data retrieval, and query optimization",
    system_prompt="""You are the Query Specialist, an expert agent responsible for database querying, data retrieval, and query optimization. Your expertise includes:

    QUERY EXECUTION:
    - Executing complex SELECT queries efficiently
    - Handling parameterized queries for security
    - Managing query results and data formatting
    - Ensuring query performance and optimization
    - Providing comprehensive query results with metadata

    QUERY OPTIMIZATION:
    - Analyzing query performance and optimization opportunities
    - Creating efficient queries that minimize resource usage
    - Understanding database indexes and query planning
    - Optimizing JOIN operations and complex queries
    - Managing query timeouts and performance monitoring

    DATA RETRIEVAL:
    - Retrieving data with proper formatting and structure
    - Handling large result sets efficiently
    - Managing data aggregation and summarization
    - Creating reports and data analysis queries
    - Ensuring data accuracy and completeness in results

    SECURITY & VALIDATION:
    - Ensuring queries are safe and secure
    - Validating query syntax and parameters
    - Preventing SQL injection and security vulnerabilities
    - Managing query permissions and access controls
    - Ensuring queries follow security best practices

    Your responsibilities:
    - Execute database queries safely and efficiently
    - Optimize query performance for best results
    - Provide comprehensive query results and analysis
    - Handle query errors and provide solutions
    - Ensure query security and data protection

    You work with expertise in SQL optimization and always ensure queries are secure, efficient, and provide accurate results.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.3,
    dynamic_temperature_enabled=True,
    tools=[query_database, get_database_schema],
)

# =============================================================================
# DATABASE DIRECTOR AGENT (COORDINATOR)
# =============================================================================
database_director_agent = Agent(
    agent_name="Database-Director",
    agent_description="Senior database director who orchestrates comprehensive database operations across all specialized teams",
    system_prompt="""You are the Database Director, the senior executive responsible for orchestrating comprehensive database operations and coordinating a team of specialized database experts. Your role is to:

    STRATEGIC COORDINATION:
    - Analyze complex database tasks and break them down into specialized operations
    - Assign tasks to the most appropriate specialist based on their unique expertise
    - Ensure comprehensive coverage of all database operations (creation, schema, data, queries)
    - Coordinate between specialists to avoid conflicts and ensure data integrity
    - Synthesize results from multiple specialists into coherent database solutions
    - Ensure all database operations align with user requirements and best practices

    TEAM LEADERSHIP:
    - Lead the Database Creator in setting up new databases and infrastructure
    - Guide the Table Manager in creating optimal table structures and schemas
    - Direct the Data Operations specialist in data insertion and update operations
    - Oversee the Query Specialist in data retrieval and analysis operations
    - Ensure all team members work collaboratively toward unified database goals
    - Provide strategic direction and feedback to optimize team performance

    DATABASE ARCHITECTURE:
    - Design comprehensive database solutions that meet user requirements
    - Ensure database operations follow best practices and standards
    - Plan database workflows that optimize performance and reliability
    - Balance immediate operational needs with long-term database health
    - Ensure database operations are secure, efficient, and maintainable
    - Optimize database operations for scalability and performance

    OPERATION ORCHESTRATION:
    - Monitor database operations across all specialists and activities
    - Analyze results to identify optimization opportunities and improvements
    - Ensure database operations deliver reliable and accurate results
    - Provide strategic recommendations based on operation outcomes
    - Coordinate complex multi-step database operations across specialists
    - Ensure continuous improvement and optimization in database management

    Your expertise includes:
    - Database architecture and design strategy
    - Team leadership and cross-functional coordination
    - Database performance analysis and optimization
    - Strategic planning and requirement analysis
    - Operation workflow management and optimization
    - Database security and best practices implementation

    You deliver comprehensive database solutions that leverage the full expertise of your specialized team, ensuring all database operations work together to provide reliable, efficient, and secure data management.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.5,
    dynamic_temperature_enabled=True,
)

# =============================================================================
# HIERARCHICAL DATABASE SWARM
# =============================================================================
# Create list of specialized database agents
database_specialists = [
    database_creator_agent,
    table_manager_agent,
    data_operations_agent,
    query_specialist_agent,
]

# Initialize the hierarchical database swarm
smart_database_swarm = HierarchicalSwarm(
    name="Smart-Database-Swarm",
    description="A comprehensive database management system with specialized agents for creation, schema management, data operations, and querying, coordinated by a database director",
    director_model_name="gpt-4.1",
    agents=database_specialists,
    max_loops=1,
    verbose=True,
)

# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATIONS
# =============================================================================
if __name__ == "__main__":
    # Configure logging
    logger.info("Starting Smart Database Swarm demonstration")

    # Example 1: Create a complete e-commerce database system
    print("=" * 80)
    print("SMART DATABASE SWARM - E-COMMERCE SYSTEM EXAMPLE")
    print("=" * 80)

    task1 = """Create a comprehensive e-commerce database system with the following requirements:
    
    1. Create a database called 'ecommerce_db' 
    2. Create tables for:
       - customers (id, name, email, phone, address, created_at)
       - products (id, name, description, price, category, stock_quantity, created_at)
       - orders (id, customer_id, order_date, total_amount, status)
       - order_items (id, order_id, product_id, quantity, unit_price)
    
    3. Insert sample data:
       - Add 3 customers
       - Add 5 products in different categories
       - Create 2 orders with multiple items
    
    4. Query the database to:
       - Show all customers with their order history
       - Display products by category with stock levels
       - Calculate total sales by product
    
    Ensure all operations are executed properly and provide comprehensive results."""

    result1 = smart_database_swarm.run(task=task1)
    print("\nE-COMMERCE DATABASE RESULT:")
    print(result1)

    # print("\n" + "=" * 80)
    # print("SMART DATABASE SWARM - EMPLOYEE MANAGEMENT SYSTEM")
    # print("=" * 80)

    # # Example 2: Employee management system
    # task2 = """Create an employee management database system:

    # 1. Create database 'company_hr'
    # 2. Create tables for:
    #    - departments (id, name, budget, manager_id)
    #    - employees (id, name, email, department_id, position, salary, hire_date)
    #    - projects (id, name, description, start_date, end_date, budget)
    #    - employee_projects (employee_id, project_id, role, hours_allocated)

    # 3. Add sample data for departments, employees, and projects
    # 4. Query for:
    #    - Employee count by department
    #    - Average salary by position
    #    - Projects with their assigned employees
    #    - Department budgets vs project allocations

    # Coordinate the team to build this system efficiently."""

    # result2 = smart_database_swarm.run(task=task2)
    # print("\nEMPLOYEE MANAGEMENT RESULT:")
    # print(result2)

    # print("\n" + "=" * 80)
    # print("SMART DATABASE SWARM - DATABASE ANALYSIS")
    # print("=" * 80)

    # # Example 3: Database analysis and optimization
    # task3 = """Analyze and optimize the existing databases:

    # 1. Get schema information for all created databases
    # 2. Analyze table structures and relationships
    # 3. Suggest optimizations for:
    #    - Index creation for better query performance
    #    - Data normalization improvements
    #    - Constraint additions for data integrity

    # 4. Update data in existing tables:
    #    - Increase product prices by 10% for electronics category
    #    - Update employee salaries based on performance criteria
    #    - Modify order statuses for completed orders

    # 5. Create comprehensive reports showing:
    #    - Database statistics and health metrics
    #    - Data distribution and patterns
    #    - Performance optimization recommendations

    # Coordinate all specialists to provide a complete database analysis."""

    # result3 = smart_database_swarm.run(task=task3)
    # print("\nDATABASE ANALYSIS RESULT:")
    # print(result3)

    # logger.info("Smart Database Swarm demonstration completed successfully")
```


- Run the file with `smart_database_swarm.py`