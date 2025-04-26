"""
Tests for the DatabaseManager components.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from text_to_sql.db.base import DatabaseManager
from text_to_sql.db.postgres import PostgresDatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Tests for the abstract DatabaseManager class."""
    
    def test_abstract_methods(self):
        """Test that we can't instantiate the abstract class."""
        with self.assertRaises(TypeError):
            DatabaseManager()
    
    def test_validate_query(self):
        """Test the default query validation logic."""
        # Create a concrete subclass for testing
        class TestManager(DatabaseManager):
            def connect(self):
                return True
            
            def disconnect(self):
                return True
            
            def execute_query(self, query, params=None):
                return [], None
            
            def get_schema(self, refresh=False):
                return {}
            
            def get_table_info(self, table_name):
                return {}
            
            def get_table_sample(self, table_name, limit=5):
                return []
        
        manager = TestManager()
        
        # Test valid SELECT query
        valid, error = manager.validate_query("SELECT * FROM users")
        self.assertTrue(valid)
        self.assertIsNone(error)
        
        # Test invalid non-SELECT query
        valid, error = manager.validate_query("INSERT INTO users (name) VALUES ('test')")
        self.assertFalse(valid)
        self.assertEqual(error, "Only SELECT queries are allowed")
        
        # Test invalid query with syntax error
        valid, error = manager.validate_query("SELECT * FROMM users")
        self.assertTrue(valid)  # Base implementation just checks if it starts with SELECT
        self.assertIsNone(error)
    
    def test_get_database_type(self):
        """Test the get_database_type method."""
        # Create a concrete subclass for testing
        class TestManager(DatabaseManager):
            def connect(self):
                return True
            
            def disconnect(self):
                return True
            
            def execute_query(self, query, params=None):
                return [], None
            
            def get_schema(self, refresh=False):
                return {}
            
            def get_table_info(self, table_name):
                return {}
            
            def get_table_sample(self, table_name, limit=5):
                return []
        
        manager = TestManager()
        self.assertEqual(manager.get_database_type(), "Test")


@pytest.mark.integration
class TestPostgresDatabaseManager(unittest.TestCase):
    """Integration tests for the PostgresDatabaseManager class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Skip if environment variables for DB connection are not set
        if not os.environ.get("TEST_DB_HOST"):
            cls.skipTest(cls, "TEST_DB_* environment variables not set")
        
        # Connection parameters
        cls.db_params = {
            "host": os.environ.get("TEST_DB_HOST", "localhost"),
            "port": int(os.environ.get("TEST_DB_PORT", "5432")),
            "dbname": os.environ.get("TEST_DB_NAME", "postgres"),
            "user": os.environ.get("TEST_DB_USER", "postgres"),
            "password": os.environ.get("TEST_DB_PASSWORD", ""),
        }
    
    def setUp(self):
        """Set up each test."""
        self.manager = PostgresDatabaseManager(self.db_params)
        self.manager.connect()
    
    def tearDown(self):
        """Clean up after each test."""
        self.manager.disconnect()
    
    def test_connect_disconnect(self):
        """Test database connection and disconnection."""
        # Initially connected in setUp
        self.assertTrue(self.manager.connected)
        
        # Disconnect
        result = self.manager.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.manager.connected)
        
        # Reconnect
        result = self.manager.connect()
        self.assertTrue(result)
        self.assertTrue(self.manager.connected)
    
    def test_execute_query(self):
        """Test query execution."""
        # Execute a simple query
        results, error = self.manager.execute_query("SELECT 1 AS test")
        
        self.assertIsNone(error)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["test"], 1)
    
    def test_get_schema(self):
        """Test schema retrieval."""
        # Get the schema
        schema = self.manager.get_schema()
        
        # Should return a dictionary
        self.assertIsInstance(schema, dict)
        
        # Force refresh
        schema_refreshed = self.manager.get_schema(refresh=True)
        self.assertIsInstance(schema_refreshed, dict)
    
    def test_validate_query(self):
        """Test query validation."""
        # Test valid query
        valid, error = self.manager.validate_query("SELECT 1 AS test")
        self.assertTrue(valid)
        self.assertIsNone(error)
        
        # Test invalid query
        valid, error = self.manager.validate_query("SELECT invalid_column FROM nonexistent_table")
        self.assertFalse(valid)
        self.assertIsNotNone(error)


class TestPostgresDatabaseManagerMocked(unittest.TestCase):
    """Unit tests for the PostgresDatabaseManager class using mocks."""
    
    def setUp(self):
        """Set up each test."""
        patcher = patch("psycopg2.pool.ThreadedConnectionPool")
        self.mock_pool = patcher.start()
        self.addCleanup(patcher.stop)
        
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        
        self.mock_pool.return_value.getconn.return_value = self.mock_connection
        self.mock_connection.cursor.return_value = self.mock_cursor
        
        self.db_params = {
            "host": "localhost",
            "port": 5432,
            "dbname": "postgres",
            "user": "postgres",
            "password": "password",
        }
        
        self.manager = PostgresDatabaseManager(self.db_params)
    
    def test_connect(self):
        """Test database connection."""
        result = self.manager.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.manager.connected)
        self.mock_pool.assert_called_once()
    
    def test_disconnect(self):
        """Test database disconnection."""
        # Connect first
        self.manager.connect()
        
        # Disconnect
        result = self.manager.disconnect()
        
        self.assertTrue(result)
        self.assertFalse(self.manager.connected)
        self.mock_pool.return_value.closeall.assert_called_once()
    
    def test_execute_query(self):
        """Test query execution."""
        # Connect first
        self.manager.connect()
        
        # Set up the mock cursor
        self.mock_cursor.fetchall.return_value = [{"test": 1}]
        
        # Execute a query
        results, error = self.manager.execute_query("SELECT 1 AS test")
        
        self.assertIsNone(error)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["test"], 1)
        
        # Check that the cursor was used correctly
        self.mock_cursor.execute.assert_called_once_with("SELECT 1 AS test", {})
        self.mock_cursor.fetchall.assert_called_once()
        self.mock_cursor.close.assert_called_once()
        
        # Check that the connection was returned to the pool
        self.mock_pool.return_value.putconn.assert_called_once_with(self.mock_connection)
    
    def test_execute_query_error(self):
        """Test query execution with an error."""
        # Connect first
        self.manager.connect()
        
        # Set up the mock cursor to raise an exception
        self.mock_cursor.execute.side_effect = Exception("Test error")
        
        # Execute a query
        results, error = self.manager.execute_query("SELECT 1 AS test")
        
        self.assertEqual(results, [])
        self.assertEqual(error, "Test error")
        
        # Check that the cursor was closed
        self.mock_cursor.close.assert_called_once()
        
        # Check that the connection was returned to the pool
        self.mock_pool.return_value.putconn.assert_called_once_with(self.mock_connection)


if __name__ == "__main__":
    unittest.main()