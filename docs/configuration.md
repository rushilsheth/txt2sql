# Configuration System

## Overview

The text-to-SQL application uses a strongly typed configuration system that provides type safety, validation, and better IDE support for configuration options. This document explains how the configuration system works and how to use it effectively.

## Configuration Classes

The system uses dataclasses to define configuration objects with proper typing:

### Database Configuration

```python
@dataclass
class DatabaseConfig:
    type: str = "postgres"
    host: str = "localhost"
    port: int = 5432
    dbname: str = "adventureworks"
    user: str = "postgres"
    password: str = ""
    min_connections: int = 1
    max_connections: int = 5
    ssl_mode: str = "prefer"
    timeout: int = 30
```

### LLM Configuration

```python
@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout: int = 30
    max_tokens: int = 1024
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
```

### Application Configuration

```python
@dataclass
class AppConfig:
    title: str = "Text-to-SQL Interface"
    description: str = "Ask questions about your data in natural language"
    theme: str = "default"
    debug_mode: bool = False
    use_semantic_engine: bool = True
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
```

### Agent Configuration

```python
@dataclass
class AgentConfig:
    max_iterations: int = 5
    auto_fix_errors: bool = True
    use_dynamic_coordinator: bool = False
    
    # Nested configurations
    query_understanding: Dict[str, Any] = field(default_factory=dict)
    schema_analysis: Dict[str, Any] = field(default_factory=dict)
    # ... more agent settings
```

### System Configuration

```python
@dataclass
class SystemConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    app: AppConfig = field(default_factory=AppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
```

## Loading Configuration

The system loads configuration from multiple sources in this order:

1. Default values defined in the dataclasses
2. Values from the config.yaml file
3. Values from environment variables

```python
# Load configuration
config = load_config("config.yaml")

# Access typed configuration
db_host = config.database.host
llm_model = config.llm.model
```

## Environment Variables

Configuration can be overridden with environment variables using this format:

```
TEXTTOSQL_SECTION_KEY=value
```

For example:
- `TEXTTOSQL_DATABASE_HOST=localhost`
- `TEXTTOSQL_LLM_MODEL=gpt-3.5-turbo`
- `TEXTTOSQL_APP_DEBUG_MODE=true`

## Using Configuration in Components

Components should accept typed configuration objects when possible:

```python
def __init__(
    self,
    db_config: Union[Dict[str, Any], DatabaseConfig],
    llm_config: Union[Dict[str, Any], LLMConfig],
    # ...
):
    # Convert dictionaries to typed configs if needed
    self.db_config = db_config if isinstance(db_config, DatabaseConfig) else DatabaseConfig.from_dict(db_config)
    self.llm_config = llm_config if isinstance(llm_config, LLMConfig) else LLMConfig.from_dict(llm_config)
```

## Adding New Configuration Options

To add new configuration options:

1. Add the option to the appropriate dataclass in `config_types.py`
2. Provide a sensible default value
3. Add the option to your config.yaml file
4. Update components to use the new option

## Configuration Best Practices

- Always provide sensible defaults
- Use type hints for all configuration options
- Document each configuration option
- Use environment variables for sensitive values like passwords and API keys
- Use specific types instead of generic `Dict[str, Any]` when possible
- Validate configuration values to fail fast if requirements are not met