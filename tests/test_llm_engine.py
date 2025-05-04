"""
Tests for the LLM Engine components.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from text_to_sql.llm.engine import LLMEngine


class TestLLMEngine(unittest.TestCase):
    """Unit tests for the LLM Engine."""
    
    def setUp(self):
        """Set up each test."""
        # Create mocks
        self.mock_db_manager = MagicMock()
        self.mock_db_manager.get_schema.return_value = {"public": {"users": {"columns": {"id": {"data_type": "integer"}}}}}
        self.mock_db_manager.get_database_type.return_value = "Test"
        self.mock_db_manager.validate_query.return_value = (True, None)
        
        # Create the LLM engine
        self.llm_engine = LLMEngine(
            model="gpt-3.5-turbo",
            db_manager=self.mock_db_manager
        )
    
    @patch("openai.chat.completions.create")
    def test_generate_sql(self, mock_openai):
        """Test SQL generation."""
        # Set up the mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "```sql\nSELECT * FROM users\n```"
        mock_openai.return_value = mock_response
        
        # Generate SQL
        sql, confidence, metadata = self.llm_engine.generate_sql_no_agents("Show me all users")
        
        # Check the results
        self.assertEqual(sql, "SELECT * FROM users")
        self.assertGreater(confidence, 0.5)
        self.assertIn("generation_time", metadata)
        self.assertEqual(metadata["model"], "gpt-3.5-turbo")
        self.assertTrue(metadata["is_valid"])
        self.assertIsNone(metadata["error"])
        
        # Check that the OpenAI API was called correctly
        mock_openai.assert_called_once()
    
    @patch("openai.chat.completions.create")
    def test_generate_sql_invalid(self, mock_openai):
        """Test SQL generation with an invalid query."""
        # Set up the mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "```sql\nSELECT * FROM nonexistent_table\n```"
        mock_openai.return_value = mock_response
        
        # Set up the mock DB manager to return an invalid query
        self.mock_db_manager.validate_query.return_value = (False, "Table does not exist")
        
        # Generate SQL
        sql, confidence, metadata = self.llm_engine.generate_sql_no_agents("Show me all data from nonexistent table")
        
        # Check the results
        self.assertEqual(sql, "SELECT * FROM nonexistent_table")
        self.assertLess(confidence, 0.5)  # Lower confidence for invalid query
        self.assertFalse(metadata["is_valid"])
        self.assertEqual(metadata["error"], "Table does not exist")
    
    @patch("openai.chat.completions.create")
    def test_extract_sql_from_response(self, mock_openai):
        """Test extracting SQL from different response formats."""
        # SQL in code block with explicit language
        sql = self.llm_engine._extract_sql_from_response("Here is the SQL:\n```sql\nSELECT * FROM users\n```")
        self.assertEqual(sql, "SELECT * FROM users")
        
        # SQL in generic code block
        sql = self.llm_engine._extract_sql_from_response("Here is the SQL:\n```\nSELECT * FROM users\n```")
        self.assertEqual(sql, "SELECT * FROM users")
        
        # SQL without code block
        sql = self.llm_engine._extract_sql_from_response("Here is the SQL:\nSELECT * FROM users")
        self.assertEqual(sql, "SELECT * FROM users")
        
        # SQL with multiple lines
        sql = self.llm_engine._extract_sql_from_response("Here is the SQL:\nSELECT *\nFROM users\nWHERE id = 1")
        self.assertEqual(sql, "SELECT * FROM users WHERE id = 1")
    
    @patch("openai.chat.completions.create")
    def test_validate_and_repair_sql(self, mock_openai):
        """Test validating and repairing SQL."""
        # Set up the mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "```sql\nSELECT * FROM users WHERE id = 1\n```"
        mock_openai.return_value = mock_response
        
        # Validate and repair SQL
        repaired_sql, was_repaired = self.llm_engine.validate_and_repair_sql(
            sql_query="SELECT * FROM users WHERE id = 1",
            natural_query="Show me user with ID 1",
            error="Invalid syntax"
        )
        
        # Check the results
        self.assertEqual(repaired_sql, "SELECT * FROM users WHERE id = 1")
        self.assertFalse(was_repaired)  # No change, so not repaired
        
        # Set up the mock OpenAI response to return a different query
        mock_response.choices[0].message.content = "```sql\nSELECT * FROM users WHERE id = 2\n```"
        mock_openai.return_value = mock_response
        
        # Validate and repair SQL
        repaired_sql, was_repaired = self.llm_engine.validate_and_repair_sql(
            sql_query="SELECT * FROM users WHERE id = 1",
            natural_query="Show me user with ID 2",
            error="Invalid syntax"
        )
        
        # Check the results
        self.assertEqual(repaired_sql, "SELECT * FROM users WHERE id = 2")
        self.assertTrue(was_repaired)  # Changed, so repaired
    
    def test_format_schema_for_prompt(self):
        """Test formatting schema for the prompt."""
        # Set up a test schema
        schema = {
            "public": {
                "users": {
                    "description": "User accounts",
                    "columns": {
                        "id": {
                            "data_type": "integer",
                            "is_primary_key": True,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "Primary key",
                            "constraint": None
                        },
                        "name": {
                            "data_type": "character varying",
                            "is_primary_key": False,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "User's name",
                            "constraint": None
                        }
                    }
                }
            }
        }
        
        # Format the schema
        formatted = self.llm_engine._format_schema_for_prompt(schema)
        
        # Check the result
        self.assertIn("Table: public.users", formatted)
        self.assertIn("Description: User accounts", formatted)
        self.assertIn("id integer PRIMARY KEY NOT NULL", formatted)
        self.assertIn("name character varying NOT NULL", formatted)
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Valid query with no error
        confidence = self.llm_engine._calculate_confidence(
            is_valid=True,
            error=None,
            response="SELECT * FROM users"
        )
        self.assertGreater(confidence, 0.5)
        
        # Invalid query with error
        confidence = self.llm_engine._calculate_confidence(
            is_valid=False,
            error="Table does not exist",
            response="SELECT * FROM nonexistent_table"
        )
        self.assertLess(confidence, 0.5)
        
        # Valid query with uncertainty in response
        confidence = self.llm_engine._calculate_confidence(
            is_valid=True,
            error=None,
            response="I'm not sure, but maybe: SELECT * FROM users"
        )
        self.assertLess(confidence, 0.7)  # Lower due to uncertainty


@pytest.mark.integration
class TestLLMEngineIntegration(unittest.TestCase):
    """Integration tests for the LLM Engine with real API calls."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Skip if environment variables for API key are not set
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            cls.skipTest(cls, "OPENAI_API_KEY environment variable not set")
        
        # Create a mock DB manager
        cls.mock_db_manager = MagicMock()
        cls.mock_db_manager.get_schema.return_value = {
            "public": {
                "users": {
                    "description": "User accounts",
                    "columns": {
                        "id": {
                            "data_type": "integer",
                            "is_primary_key": True,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "Primary key",
                            "constraint": None
                        },
                        "name": {
                            "data_type": "character varying",
                            "is_primary_key": False,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "User's name",
                            "constraint": None
                        }
                    }
                }
            }
        }
        cls.mock_db_manager.get_database_type.return_value = "Test"
        cls.mock_db_manager.validate_query.return_value = (True, None)
    
    def setUp(self):
        """Set up each test."""
        # Create the LLM engine
        self.llm_engine = LLMEngine(
            model="gpt-3.5-turbo",
            temperature=0.0,
            db_manager=self.mock_db_manager
        )
    
    @pytest.mark.skipif(not bool(os.environ.get("OPENAI_API_KEY")), reason="OPENAI_API_KEY not set")
    def test_validate_and_repair_sql_integration(self):
        """Test validating and repairing SQL with the real API."""
        # Validate and repair SQL
        repaired_sql, was_repaired = self.llm_engine.validate_and_repair_sql(
            sql_query="SELECT * FROM user",  # Incorrect table name (should be users)
            natural_query="Show me all users",
            error="Table 'user' does not exist"
        )
        
        # Check the results
        self.assertIn("SELECT", repaired_sql.upper())
        self.assertIn("FROM", repaired_sql.upper())
        self.assertIn("users", repaired_sql.lower())
        self.assertTrue(was_repaired)