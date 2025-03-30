import pytest
from unittest.mock import Mock, patch
from swarms.utils.typedb_wrapper import TypeDBWrapper, TypeDBConfig

@pytest.fixture
def mock_typedb():
    """Mock TypeDB client and session."""
    with patch('swarms.utils.typedb_wrapper.TypeDB') as mock_typedb:
        mock_client = Mock()
        mock_session = Mock()
        mock_typedb.core_client.return_value = mock_client
        mock_client.session.return_value = mock_session
        yield mock_typedb, mock_client, mock_session

@pytest.fixture
def typedb_wrapper(mock_typedb):
    """Create a TypeDBWrapper instance with mocked dependencies."""
    config = TypeDBConfig(
        uri="test:1729",
        database="test_db",
        username="test_user",
        password="test_pass"
    )
    return TypeDBWrapper(config)

def test_initialization(typedb_wrapper):
    """Test TypeDBWrapper initialization."""
    assert typedb_wrapper.config.uri == "test:1729"
    assert typedb_wrapper.config.database == "test_db"
    assert typedb_wrapper.config.username == "test_user"
    assert typedb_wrapper.config.password == "test_pass"

def test_connect(typedb_wrapper, mock_typedb):
    """Test connection to TypeDB."""
    mock_typedb, mock_client, mock_session = mock_typedb
    typedb_wrapper._connect()
    
    mock_typedb.core_client.assert_called_once_with("test:1729")
    mock_client.session.assert_called_once_with(
        "test_db",
        "DATA",
        "test_user",
        "test_pass"
    )

def test_define_schema(typedb_wrapper, mock_typedb):
    """Test schema definition."""
    mock_typedb, mock_client, mock_session = mock_typedb
    schema = "define person sub entity;"
    
    with patch.object(typedb_wrapper.session, 'transaction') as mock_transaction:
        mock_transaction.return_value.__enter__.return_value.query.define.return_value = None
        typedb_wrapper.define_schema(schema)
        
        mock_transaction.assert_called_once_with("WRITE")
        mock_transaction.return_value.__enter__.return_value.query.define.assert_called_once_with(schema)

def test_insert_data(typedb_wrapper, mock_typedb):
    """Test data insertion."""
    mock_typedb, mock_client, mock_session = mock_typedb
    query = "insert $p isa person;"
    
    with patch.object(typedb_wrapper.session, 'transaction') as mock_transaction:
        mock_transaction.return_value.__enter__.return_value.query.insert.return_value = None
        typedb_wrapper.insert_data(query)
        
        mock_transaction.assert_called_once_with("WRITE")
        mock_transaction.return_value.__enter__.return_value.query.insert.assert_called_once_with(query)

def test_query_data(typedb_wrapper, mock_typedb):
    """Test data querying."""
    mock_typedb, mock_client, mock_session = mock_typedb
    query = "match $p isa person; get;"
    mock_result = [Mock()]
    
    with patch.object(typedb_wrapper.session, 'transaction') as mock_transaction:
        mock_transaction.return_value.__enter__.return_value.query.get.return_value = mock_result
        result = typedb_wrapper.query_data(query)
        
        mock_transaction.assert_called_once_with("READ")
        mock_transaction.return_value.__enter__.return_value.query.get.assert_called_once_with(query)
        assert len(result) == 1

def test_delete_data(typedb_wrapper, mock_typedb):
    """Test data deletion."""
    mock_typedb, mock_client, mock_session = mock_typedb
    query = "match $p isa person; delete $p;"
    
    with patch.object(typedb_wrapper.session, 'transaction') as mock_transaction:
        mock_transaction.return_value.__enter__.return_value.query.delete.return_value = None
        typedb_wrapper.delete_data(query)
        
        mock_transaction.assert_called_once_with("WRITE")
        mock_transaction.return_value.__enter__.return_value.query.delete.assert_called_once_with(query)

def test_close(typedb_wrapper, mock_typedb):
    """Test connection closing."""
    mock_typedb, mock_client, mock_session = mock_typedb
    typedb_wrapper.close()
    
    mock_session.close.assert_called_once()
    mock_client.close.assert_called_once()

def test_context_manager(typedb_wrapper, mock_typedb):
    """Test context manager functionality."""
    mock_typedb, mock_client, mock_session = mock_typedb
    
    with typedb_wrapper as db:
        assert db == typedb_wrapper
    
    mock_session.close.assert_called_once()
    mock_client.close.assert_called_once()

def test_error_handling(typedb_wrapper, mock_typedb):
    """Test error handling."""
    mock_typedb, mock_client, mock_session = mock_typedb
    
    # Test connection error
    mock_typedb.core_client.side_effect = Exception("Connection failed")
    with pytest.raises(Exception) as exc_info:
        typedb_wrapper._connect()
    assert "Connection failed" in str(exc_info.value)
    
    # Test query error
    with patch.object(typedb_wrapper.session, 'transaction') as mock_transaction:
        mock_transaction.return_value.__enter__.return_value.query.get.side_effect = Exception("Query failed")
        with pytest.raises(Exception) as exc_info:
            typedb_wrapper.query_data("test query")
        assert "Query failed" in str(exc_info.value) 