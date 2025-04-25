### Structured Logging

The system includes comprehensive structured logging for better integration with monitoring tools:

- **JSON Formatting**: Logs can be output as JSON for easy parsing
- **Tagging System**: Tags categorize logs for filtering and analysis
- **Structured Data**: Additional context included in structured format
- **Integration Support**: Ready for Splunk, Sentry, and ELK stack

Example usage:

```python
logger.info(
    "Query processed successfully",
    tags={"component": "processor", "action": "complete", "user_id": "12345"},
    structured_data={
        "query_id": "abc123",
        "processing_time_ms": 235,
        "tokens_used": 512
    }
)
```

Configure structured logging in `config.yaml`:

```yaml
logging:
  structured: true    # Enable structured logging
  json: true          # Output as JSON in console
  json_file: true     # Output as JSON in log file
```

For more details, see the [structured logging documentation](docs/examples/structured_logging.md).## Project Structure

The project follows a modern Python package structure:

```
text-to-sql/
├── src/                   # Source code directory
│   └── text_to_sql/       # Main package
│   └── scripts/           # Utility scripts
├── tests/                 # Test directory
├── docs/                  # Documentation
├── data/                  # Sample data
├── config.yaml            # Configuration file
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── pyproject.toml         # Poetry configuration
└── Makefile               # Make commands
```

## Enhanced Features

### Caching System

The application includes a sophisticated caching system for improved performance:

- **Schema Caching**: Database schemas are cached to avoid redundant introspection
- **Query Caching**: SQL query results are cached to speed up repeated queries
- **LLM Caching**: LLM responses are cached to reduce API usage and costs

### Advanced Logging

Enhanced logging capabilities provide better visibility:

- **Configurable Log Levels**: Set log levels per component
- **Colored Console Output**: Improved readability with colored logs
- **File Logging**: Log to files with automatic rotation
- **Custom Log Levels**: Added TRACE level for detailed debugging

### Environment Validation

The application validates required environment variables and prerequisites:

- **OpenAI API Key**: Validated for agent-based approaches
- **Database Connection**: Tested before application start
- **Feature-Specific Validation**: Only checks prerequisites for enabled features

### Containerization

The project can be run in Docker containers:

```bash
# Build and start with Docker Compose
docker-compose up -d

# Run with dynamic coordinator
docker-compose run -e TEXTTOSQL_AGENT_USE_DYNAMIC_COORDINATOR=true app

# Access the application
# http://localhost:7860
```## Agent-based Architecture

The project implements a sophisticated agent-based architecture inspired by Google's A2A (architecture-to-architecture) approach but without relying on LangChain. This provides a more transparent, controllable, and extensible solution.

### Agent Structure

The agent system consists of several specialized agents:

1. **Query Understanding Agent** - Extracts intent and entities from natural language queries
2. **Schema Analysis Agent** - Identifies relevant tables and columns for a query
3. **SQL Generation Agent** - Converts natural language to SQL based on the schema
4. **Query Validation Agent** - Validates and fixes SQL queries before execution
5. **Result Explanation Agent** - Provides natural language explanations of query results
6. **Visualization Agent** - Suggests appropriate visualizations for the results
7. **Coordinator Agent** - Orchestrates the other agents in a coherent pipeline

### Agent Context

Agents share a common context object that maintains the state of the query processing:

```python
@dataclass
class AgentContext:
    # Original user input
    user_query: str = ""
    
    # Extracted intent and entities
    query_intent: str = ""
    query_entities: List[str] = field(default_factory=list)
    
    # Generated SQL
    sql_query: str = ""
    sql_params: Dict[str, Any] = field(default_factory=dict)
    
    # Query execution results
    query_results: List[Dict[str, Any]] = field(default_factory=list)
    result_error: Optional[str] = None
    
    # Agent traces and explanations
    reasoning_steps: List[str] = field(default_factory=list)
    explanations: Dict[str, str] = field(default_factory=dict)
    
    # And more...
```

## Coordinator Types

The system supports two types of agent coordinators:

1. **Simple Coordinator** (default) - Uses a fixed pipeline of agent invocations
2. **Dynamic Coordinator** - Uses OpenAI function calling to decide which agent to invoke next

You can run the system with either coordinator:

```bash
# Use simple coordinator (default)
make run

# Use dynamic coordinator 
make run dynamic
# or manually:
# poetry run python -m text_to_sql --dynamic

# Use standard (non-agent) approach
make run standard
# or manually:
# poetry run python -m text_to_sql --standard

# Enable debug mode
make run debug
# or manually:
# poetry run python -m text_to_sql --debug
```

### Dynamic Coordinator Features

The dynamic coordinator provides enhanced capabilities:

- **Contextual Decision-Making**: Uses the current state to decide the next best agent to invoke
- **Self-Reflection**: Periodically reflects on progress and adjusts strategy
- **Error Recovery**: Can change course when encountering issues
- **Customized Pipelines**: Creates tailored processing sequences for each query

When using the dynamic coordinator, you'll see its reasoning process in the debug view, showing why it chose each agent.

### Configuring Agents

Agent behavior can be configured in the `config.yaml` file under the `agent` section:

```yaml
agent:
  max_iterations: 5
  auto_fix_errors: true
  query_understanding:
    use_examples: true
    max_tokens: 512
  schema_analysis:
    table_limit: 10
    column_limit: 20
  # ... more agent-specific settings
```# Text-to-SQL: Natural Language Database Interface

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

## Example Usage

Once the application is running, you can ask natural language questions about the AdventureWorks data:

- "How many products are in each category?"
- "What's the average list price of products by category?"
- "Show me the total sales amount by month for 2014"
- "Which customers have spent the most in the last year?"
- "List the employees who have processed the most orders"

The application will convert these questions to SQL, execute the queries, and display the results with appropriate visualizations.

## Architecture

The project follows a modular architecture with three main components:

1. **Database Manager**: Handles database connections, schema introspection, and query execution
2. **LLM Engine**: Processes natural language queries and generates SQL
3. **Visualization Layer**: Presents query results and provides interactive filtering

For more details, see the [architecture documentation](docs/ARCHITECTURE.md).

## Extending the Database Support

To add support for a new database backend, extend the `DatabaseManager` base class:

```python
from text_to_sql.db.base import DatabaseManager

class MySQLManager(DatabaseManager):
    """MySQL implementation of the DatabaseManager."""
    
    def connect(self, connection_string):
        # Implementation for MySQL connection
        pass
        
    def execute_query(self, query):
        # MySQL-specific query execution
        pass
        
    def get_schema(self):
        # MySQL schema introspection
        pass
```

## Configuration

The application can be configured using a YAML file or environment variables. See `config.yaml` for available options.

Environment variables use the format `TEXTTOSQL_SECTION_KEY`. For example:
- `TEXTTOSQL_DATABASE_HOST` - Database host
- `TEXTTOSQL_LLM_MODEL` - LLM model to use
- `TEXTTOSQL_APP_PORT` - Port for the web interface

## Project Structure

```
text-to-sql/
├── pyproject.toml         # Poetry configuration
├── Makefile               # Make commands for common operations
├── README.md              # Project documentation
├── config.yaml            # Configuration file
├── scripts/               # Utility scripts
│   ├── setup_db.py        # Script to set up the database
│   └── download_data.py   # Script to download sample data
├── text_to_sql/           # Main package
│   ├── __init__.py
│   ├── __main__.py        # CLI entry point
│   ├── config.py          # Configuration settings
│   ├── db/                # Database management
│   │   ├── __init__.py
│   │   ├── base.py        # Abstract base class for DB connections
│   │   ├── postgres.py    # PostgreSQL implementation
│   │   └── schema.py      # Schema handling utilities
│   ├── llm/               # LLM integration
│   │   ├── __init__.py
│   │   ├── engine.py      # Core LLM functionality
│   │   ├── prompts.py     # Prompt templates
│   │   └── semantic.py    # Semantic understanding components
│   └── visualization/     # Visualization components
│       ├── __init__.py
│       ├── app.py         # Gradio application
│       ├── charts.py      # Chart components
│       └── dashboard.py   # Dashboard layout
└── data/                  # Sample data and resources
    └── adventureworks.sql # AdventureWorks SQL file (downloaded)
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

        
    def execute_query(self, query):
        # MySQL-specific query execution
        pass
        
    def get_schema(self):
        # MySQL schema introspection
        pass
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