# Text-to-SQL: Natural Language Database Interface

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A robust natural language interface for SQL databases that handles schema changes, provides semantic understanding, and visualizes query results.

## Features

- 🔍 **Natural Language Queries**: Convert plain English questions into SQL queries
- 🔄 **Database Abstraction**: Support for PostgreSQL with extensibility for other database backends
- 🧠 **Semantic Understanding**: Intelligent parsing of user intent and mapping to database schema
- 📊 **Interactive Visualization**: Gradio-based dashboard for query results
- 🔃 **Schema Adaptation**: Automatically adapts to schema changes in the database

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 15 or higher, including the psql client
  - On macOS, you can install the psql client via Homebrew:
    ```
    brew install libpq
    brew link --force libpq
    echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc
    ```
    Alternatively, add `/opt/homebrew/opt/libpq/bin` to your PATH.
- Docker
  - Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- Make (optional, for convenience commands)
- Poetry for dependency management

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/text-to-sql.git
cd text-to-sql
```

2. Set up the environment using Poetry:

```bash
make setup
# or manually:
# poetry install
```

3. Download the AdventureWorks sample database:

```bash
make download-data
# or manually:
# poetry run python src/text_to_sql/scripts/download_data.py
```

4. Configure the database connection:

Edit `config.yaml` to set your database connection parameters, or use environment variables:

```bash
export TEXTTOSQL_DATABASE_HOST=localhost
export TEXTTOSQL_DATABASE_PORT=5432
export TEXTTOSQL_DATABASE_USER=postgres
export TEXTTOSQL_DATABASE_PASSWORD=yourpassword
```

5. Set up the database:

```bash
make setup-db
# or manually:
# poetry run python src/text_to_sql/scripts/setup_db.py
```

6. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key
```

7. Run the application:

```bash
make run
# or manually:
# poetry run python -m text_to_sql
```

## Execution Modes

The application can be run in different modes:

```bash
# Default agent-based approach with simple coordinator
make run

# Standard (non-agent) approach
make standard

# Agent-based approach with dynamic coordinator
make dynamic

# Debug mode
make debug

# Dynamic coordinator with debug mode
make dynamic-debug
```

## Example Usage

Once the application is running, you can ask natural language questions about the AdventureWorks data:

- "How many products are in each category?"
- "What's the average list price of products by category?"
- "Show me the total sales amount by month for 2014"
- "Which customers have spent the most in the last year?"
- "List the employees who have processed the most orders"

The application will convert these questions to SQL, execute the queries, and display the results with appropriate visualizations.

## Key Components

### Agent-based Architecture

The system uses a sophisticated agent-based architecture inspired by Google's A2A approach:

- **Specialized Agents**: Query Understanding, Schema Analysis, SQL Generation, etc.
- **Agent Coordination**: Fixed pipeline or dynamic decision-making
- **Context Sharing**: Shared state between agents with full traceability

See the [Agent System Documentation](docs/agent_systems.md) for details.

### Type-safe Configuration

Strongly-typed configuration system with environment variable support:

```python
@dataclass
class DatabaseConfig:
    type: str = "postgres"
    host: str = "localhost"
    # ...more fields with proper types
```

See the [Configuration Documentation](docs/configuration.md) for details.

### Structured Logging

Comprehensive structured logging for better monitoring integration:

```python
logger.info(
    "Query processed",
    tags={"component": "agent", "action": "process"},
    structured_data={"query_id": "abc123", "processing_time_ms": 235}
)
```

See the [Structured Logging Documentation](docs/structured_logging.md) for details.

### Enhanced Features

- **Caching System**: Schema, query results, and LLM responses are cached
- **Database Abstraction**: Support for multiple database backends
- **Containerization**: Docker support for easy deployment
- **Visualization**: Automatic chart selection based on data structure

## Project Structure

```
text-to-sql/
├── src/                   # Source code directory
│   └── text_to_sql/       # Main package
├── tests/                 # Test directory
├── docs/                  # Documentation
├── data/                  # Sample data
├── config.yaml            # Configuration file
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── pyproject.toml         # Poetry configuration
└── Makefile               # Make commands
```

## Extending the Database Support

To add support for a new database backend, extend the `DatabaseManager` base class:

```python
from text_to_sql.db.base import DatabaseManager

class MySQLManager(DatabaseManager):
    """MySQL implementation of the DatabaseManager."""
    
    def connect(self):
        # Implementation for MySQL connection
        pass
        
    def execute_query(self, query, params=None):
        # MySQL-specific query execution
        pass
        
    def get_schema(self, refresh=False):
        # MySQL schema introspection
        pass
```

## Docker Deployment

Run the application with Docker:

```bash
# Build and start with Docker Compose
docker-compose up -d

# Run with dynamic coordinator
docker-compose run -e TEXTTOSQL_AGENT_USE_DYNAMIC_COORDINATOR=true app

# Access the application
# http://localhost:7860
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AdventureWorks sample database by Microsoft
- Inspired by various text-to-SQL research projects
- Built with the Google A2A approach in mind