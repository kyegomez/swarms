# Smart Database Swarm

A fully autonomous database management system powered by hierarchical multi-agent workflow using the Swarms framework.

## Overview

The Smart Database Swarm is an intelligent database management system that uses specialized AI agents to handle different aspects of database operations. The system follows a hierarchical architecture where a Database Director coordinates specialized worker agents to execute complex database tasks.

## Architecture

### Hierarchical Structure

```
Database Director (Coordinator)
├── Database Creator (Creates databases)
├── Table Manager (Manages table schemas)
├── Data Operations (Handles data insertion/updates)
└── Query Specialist (Executes queries and retrieval)
```

### Agent Specializations

1. **Database Director**: Orchestrates all database operations and coordinates specialist agents
2. **Database Creator**: Specializes in creating and initializing databases
3. **Table Manager**: Expert in table creation, schema design, and structure management
4. **Data Operations**: Handles data insertion, updates, and manipulation
5. **Query Specialist**: Manages database queries, data retrieval, and optimization

## Features

- **Autonomous Database Management**: Complete database lifecycle management
- **Intelligent Task Distribution**: Automatic assignment of tasks to appropriate specialists
- **Schema Validation**: Ensures proper table structures and data integrity
- **Security**: Built-in SQL injection prevention and query validation
- **Performance Optimization**: Query optimization and efficient data operations
- **Comprehensive Error Handling**: Robust error management and reporting
- **Multi-format Data Support**: JSON-based data insertion and flexible query parameters

## Database Tools

### Core Functions

1. **`create_database(database_name, database_path)`**: Creates new SQLite databases
2. **`create_table(database_path, table_name, schema)`**: Creates tables with specified schemas
3. **`insert_data(database_path, table_name, data)`**: Inserts data into tables
4. **`query_database(database_path, query, params)`**: Executes SELECT queries
5. **`update_table_data(database_path, table_name, update_data, where_clause)`**: Updates existing data
6. **`get_database_schema(database_path)`**: Retrieves comprehensive schema information

## Usage Examples

### Basic Usage

```python
from smart_database_swarm import smart_database_swarm

# Simple database creation and setup
task = """
Create a user management database:
1. Create database 'user_system'
2. Create users table with id, username, email, created_at
3. Insert 5 sample users
4. Query all users ordered by creation date
"""

result = smart_database_swarm.run(task=task)
print(result)
```

### E-commerce System

```python
# Complex e-commerce database system
ecommerce_task = """
Create a comprehensive e-commerce database system:

1. Create database 'ecommerce_store'
2. Create tables:
   - customers (id, name, email, phone, address, created_at)
   - products (id, name, description, price, category, stock, created_at)
   - orders (id, customer_id, order_date, total_amount, status)
   - order_items (id, order_id, product_id, quantity, unit_price)

3. Insert sample data:
   - 10 customers with realistic information
   - 20 products across different categories
   - 15 orders with multiple items each

4. Execute analytical queries:
   - Top selling products by quantity
   - Customer lifetime value analysis
   - Monthly sales trends
   - Inventory levels by category
"""

result = smart_database_swarm.run(task=ecommerce_task)
```

### Data Analysis and Reporting

```python
# Advanced data analysis
analysis_task = """
Analyze the existing databases and provide insights:

1. Get schema information for all databases
2. Generate data quality reports
3. Identify optimization opportunities
4. Create performance metrics dashboard
5. Suggest database improvements

Query patterns:
- Customer segmentation analysis
- Product performance metrics
- Order fulfillment statistics
- Revenue analysis by time periods
"""

result = smart_database_swarm.run(task=analysis_task)
```

## Data Formats

### Table Schema Definition

```python
# Column definitions with types and constraints
schema = "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
```

### Data Insertion Formats

#### Format 1: List of Dictionaries
```json
[
  {"name": "John Doe", "email": "john@example.com"},
  {"name": "Jane Smith", "email": "jane@example.com"}
]
```

#### Format 2: Columns and Values
```json
{
  "columns": ["name", "email"],
  "values": [
    ["John Doe", "john@example.com"],
    ["Jane Smith", "jane@example.com"]
  ]
}
```

### Update Operations

```json
{
  "salary": 75000,
  "department": "Engineering",
  "last_updated": "2024-01-15"
}
```

## Advanced Features

### Security

- **SQL Injection Prevention**: Parameterized queries and input validation
- **Query Validation**: Only SELECT queries allowed for query operations
- **Input Sanitization**: Automatic cleaning and validation of inputs

### Performance

- **Connection Management**: Efficient database connection handling
- **Query Optimization**: Intelligent query planning and execution
- **Batch Operations**: Support for bulk data operations

### Error Handling

- **Comprehensive Error Messages**: Detailed error reporting and solutions
- **Graceful Degradation**: System continues operating despite individual failures
- **Transaction Safety**: Atomic operations with rollback capabilities

## Best Practices

### Database Design

1. **Use Proper Data Types**: Choose appropriate SQL data types for your data
2. **Implement Constraints**: Use PRIMARY KEY, FOREIGN KEY, and CHECK constraints
3. **Normalize Data**: Follow database normalization principles
4. **Index Strategy**: Create indexes for frequently queried columns

### Agent Coordination

1. **Clear Task Definitions**: Provide specific, actionable task descriptions
2. **Sequential Operations**: Allow agents to complete dependencies before next steps
3. **Comprehensive Requirements**: Include all necessary details in task descriptions
4. **Result Validation**: Review agent outputs for completeness and accuracy

### Data Operations

1. **Backup Before Updates**: Always backup data before major modifications
2. **Test Queries**: Validate queries on sample data before production execution
3. **Monitor Performance**: Track query execution times and optimize as needed
4. **Validate Data**: Ensure data integrity through proper validation

## File Structure

```
examples/guides/smart_database/
├── smart_database_swarm.py    # Main implementation
├── README.md                  # This documentation
└── databases/                 # Generated databases (auto-created)
```

## Dependencies

- `swarms`: Core framework for multi-agent systems
- `sqlite3`: Database operations (built-in Python)
- `json`: Data serialization (built-in Python)
- `pathlib`: File path operations (built-in Python)
- `loguru`: Minimal logging functionality

## Running the System

```bash
# Navigate to the smart_database directory
cd examples/guides/smart_database

# Run the demonstration
python smart_database_swarm.py

# The system will create databases in ./databases/ directory
# Check the generated databases and results
```

## Expected Output

The system will create:

1. **Databases**: SQLite database files in `./databases/` directory
2. **Detailed Results**: JSON-formatted operation results
3. **Agent Coordination**: Logs showing how tasks are distributed
4. **Performance Metrics**: Execution times and success statistics

## Troubleshooting

### Common Issues

1. **Database Not Found**: Ensure database path is correct and accessible
2. **Schema Errors**: Verify SQL syntax in table creation statements
3. **Data Format Issues**: Check JSON formatting for data insertion
4. **Permission Errors**: Ensure write permissions for database directory

### Debug Mode

Enable verbose logging to see detailed agent interactions:

```python
smart_database_swarm.verbose = True
result = smart_database_swarm.run(task=your_task)
```

## Contributing

To extend the Smart Database Swarm:

1. **Add New Tools**: Create additional database operation functions
2. **Enhance Agents**: Improve agent prompts and capabilities
3. **Add Database Types**: Support for PostgreSQL, MySQL, etc.
4. **Performance Optimization**: Implement caching and connection pooling

## License

This project is part of the Swarms framework and follows the same licensing terms.
