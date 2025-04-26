"""
Tests for the Schema Analyzer components.
"""

import unittest
from unittest.mock import MagicMock

from text_to_sql.db.schema import SchemaAnalyzer


class TestSchemaAnalyzer(unittest.TestCase):
    """Unit tests for the SchemaAnalyzer."""
    
    def setUp(self):
        """Set up each test."""
        # Create a test schema
        self.schema = {
            "public": {
                "customers": {
                    "description": "Customer accounts",
                    "columns": {
                        "customer_id": {
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
                            "description": "Customer name",
                            "constraint": None
                        }
                    }
                },
                "orders": {
                    "description": "Customer orders",
                    "columns": {
                        "order_id": {
                            "data_type": "integer",
                            "is_primary_key": True,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "Primary key",
                            "constraint": None
                        },
                        "customer_id": {
                            "data_type": "integer",
                            "is_primary_key": False,
                            "is_foreign_key": True,
                            "not_null": True,
                            "description": "Foreign key to customers",
                            "constraint": "REFERENCES public.customers(customer_id)"
                        },
                        "order_date": {
                            "data_type": "date",
                            "is_primary_key": False,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "Order date",
                            "constraint": None
                        }
                    }
                },
                "order_items": {
                    "description": "Order items",
                    "columns": {
                        "item_id": {
                            "data_type": "integer",
                            "is_primary_key": True,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "Primary key",
                            "constraint": None
                        },
                        "order_id": {
                            "data_type": "integer",
                            "is_primary_key": False,
                            "is_foreign_key": True,
                            "not_null": True,
                            "description": "Foreign key to orders",
                            "constraint": "REFERENCES public.orders(order_id)"
                        },
                        "product_id": {
                            "data_type": "integer",
                            "is_primary_key": False,
                            "is_foreign_key": True,
                            "not_null": True,
                            "description": "Foreign key to products",
                            "constraint": "REFERENCES public.products(product_id)"
                        },
                        "quantity": {
                            "data_type": "integer",
                            "is_primary_key": False,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "Item quantity",
                            "constraint": None
                        }
                    }
                },
                "products": {
                    "description": "Product catalog",
                    "columns": {
                        "product_id": {
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
                            "description": "Product name",
                            "constraint": None
                        },
                        "price": {
                            "data_type": "numeric",
                            "is_primary_key": False,
                            "is_foreign_key": False,
                            "not_null": True,
                            "description": "Product price",
                            "constraint": None
                        },
                        "category_id": {
                            "data_type": "integer",
                            "is_primary_key": False,
                            "is_foreign_key": True,
                            "not_null": True,
                            "description": "Foreign key to categories",
                            "constraint": "REFERENCES public.categories(category_id)"
                        }
                    }
                },
                "categories": {
                    "description": "Product categories",
                    "columns": {
                        "category_id": {
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
                            "description": "Category name",
                            "constraint": None
                        }
                    }
                }
            }
        }
        
        # Create the analyzer
        self.analyzer = SchemaAnalyzer(self.schema)
    
    def test_extract_relationships(self):
        """Test extracting relationships from the schema."""
        # Check that relationships were extracted
        self.assertGreater(len(self.analyzer.relationships), 0)
        
        # Check for specific relationships
        orders_to_customers = None
        for rel in self.analyzer.relationships:
            if rel['source_table'] == 'orders' and rel['target_table'] == 'public.customers':
                orders_to_customers = rel
                break
        
        self.assertIsNotNone(orders_to_customers)
        self.assertEqual(orders_to_customers['source_column'], 'customer_id')
        self.assertEqual(orders_to_customers['relation_type'], 'many-to-one')
    
    def test_identify_primary_entities(self):
        """Test identifying primary entities."""
        # Check that primary entities were identified
        self.assertGreater(len(self.analyzer.primary_entities), 0)
        
        # Check for specific entities
        # Customers should be a primary entity (referenced by orders)
        customers_entity = None
        for entity in self.analyzer.primary_entities:
            if entity['table'] == 'public.customers':
                customers_entity = entity
                break
        
        self.assertIsNotNone(customers_entity)
        self.assertGreater(customers_entity['reference_count'], 0)
    
    def test_identify_reference_tables(self):
        """Test identifying reference tables."""
        # Check that reference tables were identified
        self.assertGreater(len(self.analyzer.reference_tables), 0)
        
        # Categories should be a reference table
        categories_table = None
        for table in self.analyzer.reference_tables:
            if table['table'] == 'categories':
                categories_table = table
                break
        
        self.assertIsNotNone(categories_table)
        self.assertEqual(categories_table['schema'], 'public')
    
    def test_get_table_relationships(self):
        """Test getting relationships for a specific table."""
        # Get relationships for orders
        relationships = self.analyzer.get_table_relationships('orders')
        
        # Check that relationships were found
        self.assertGreater(len(relationships), 0)
        
        # Check for specific relationships
        orders_to_customers = None
        for rel in relationships:
            if rel['source_table'] == 'orders' and rel['target_table'] == 'public.customers':
                orders_to_customers = rel
                break
        
        self.assertIsNotNone(orders_to_customers)
    
    def test_get_join_path(self):
        """Test getting a join path between tables."""
        # Get join path from customers to products
        path = self.analyzer.get_join_path('customers', 'products')
        
        # Check that a path was found
        self.assertGreater(len(path), 0)
        
        # Path should go through orders and order_items
        tables_in_path = [rel['source_table'] for rel in path] + [path[-1]['target_table']]
        self.assertIn('customers', tables_in_path)
        self.assertIn('orders', tables_in_path)
        self.assertIn('order_items', tables_in_path)
        self.assertIn('products', tables_in_path)
    
    def test_suggest_query_tables(self):
        """Test suggesting tables for a query."""
        # Suggest tables for a query about customers
        suggested_tables = self.analyzer.suggest_query_tables(['customer', 'name'])
        
        # Check that suggestions were made
        self.assertGreater(len(suggested_tables), 0)
        
        # Customers table should be suggested
        self.assertTrue(any('customers' in table for table in suggested_tables))
    
    def test_get_column_samples(self):
        """Test getting column samples."""
        # Create a mock database manager
        mock_db_manager = MagicMock()
        mock_db_manager.execute_query.return_value = (
            [{'name': 'Customer 1'}, {'name': 'Customer 2'}],
            None
        )
        
        # Get column samples for customers
        samples = self.analyzer.get_column_samples('customers', mock_db_manager)
        
        # Check that samples were returned
        self.assertIn('name', samples)
        self.assertEqual(len(samples['name']), 2)
        self.assertEqual(samples['name'][0], 'Customer 1')
        self.assertEqual(samples['name'][1], 'Customer 2')
        
        # Check that the execute_query method was called correctly
        mock_db_manager.execute_query.assert_called_once()
        call_args = mock_db_manager.execute_query.call_args[0]
        self.assertIn('SELECT DISTINCT name', call_args[0])
        self.assertIn('public.customers', call_args[0])


if __name__ == "__main__":
    unittest.main()