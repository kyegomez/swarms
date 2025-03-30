from swarms.utils.typedb_wrapper import TypeDBWrapper, TypeDBConfig

def main():
    # Initialize TypeDB wrapper with custom configuration
    config = TypeDBConfig(
        uri="localhost:1729",
        database="swarms_example",
        username="admin",
        password="password"
    )
    
    # Define schema for a simple knowledge graph
    schema = """
    define
    person sub entity,
        owns name: string,
        owns age: long,
        plays role;
    
    role sub entity,
        owns title: string,
        owns department: string;
    
    works_at sub relation,
        relates person,
        relates role;
    """
    
    # Example data insertion
    insert_queries = [
        """
        insert
        $p isa person, has name "John Doe", has age 30;
        $r isa role, has title "Software Engineer", has department "Engineering";
        (person: $p, role: $r) isa works_at;
        """,
        """
        insert
        $p isa person, has name "Jane Smith", has age 28;
        $r isa role, has title "Data Scientist", has department "Data Science";
        (person: $p, role: $r) isa works_at;
        """
    ]
    
    # Example queries
    query_queries = [
        # Get all people
        "match $p isa person; get;",
        
        # Get people in Engineering department
        """
        match
        $p isa person;
        $r isa role, has department "Engineering";
        (person: $p, role: $r) isa works_at;
        get $p;
        """,
        
        # Get people with their roles
        """
        match
        $p isa person, has name $n;
        $r isa role, has title $t;
        (person: $p, role: $r) isa works_at;
        get $n, $t;
        """
    ]
    
    try:
        with TypeDBWrapper(config) as db:
            # Define schema
            print("Defining schema...")
            db.define_schema(schema)
            
            # Insert data
            print("\nInserting data...")
            for query in insert_queries:
                db.insert_data(query)
            
            # Query data
            print("\nQuerying data...")
            for i, query in enumerate(query_queries, 1):
                print(f"\nQuery {i}:")
                results = db.query_data(query)
                print(f"Results: {results}")
            
            # Example of deleting data
            print("\nDeleting data...")
            delete_query = """
            match
            $p isa person, has name "John Doe";
            delete $p;
            """
            db.delete_data(delete_query)
            
            # Verify deletion
            print("\nVerifying deletion...")
            verify_query = "match $p isa person, has name $n; get $n;"
            results = db.query_data(verify_query)
            print(f"Remaining people: {results}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 