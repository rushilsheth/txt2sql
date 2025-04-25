# Structured Logging Examples

The text-to-SQL system supports structured logging, which makes it easier to integrate with monitoring systems like Sentry, Splunk, or ELK stack. This document shows how to use structured logging effectively in your code.

## Basic Usage

```python
from text_to_sql.utils.logging import get_logger

# Get a logger for your component
logger = get_logger("text_to_sql.my_component")

# Basic logging with tags
logger.info(
    "Processing query", 
    tags={"component": "processor", "action": "process", "user_id": "12345"}
)

# Log with both tags and structured data
logger.info(
    "Query processed successfully",
    tags={"component": "processor", "action": "complete", "user_id": "12345"},
    structured_data={
        "query_id": "abc123",
        "processing_time_ms": 235,
        "tokens_used": 512
    }
)

# Error logging with structured information
try:
    # Some operation that might fail
    result = process_query(query)
except Exception as e:
    logger.exception(
        "Error processing query",
        tags={"component": "processor", "action": "error", "error_type": type(e).__name__},
        structured_data={
            "query_id": "abc123",
            "query_text": query[:100]  # Truncated for brevity
        }
    )
```

## JSON Output

When configured to output as JSON, the logs will have a format suitable for ingestion by monitoring systems:

```json
{
  "timestamp": "2025-04-26T12:34:56.789",
  "level": "INFO",
  "logger": "text_to_sql.my_component",
  "message": "Query processed successfully",
  "pathname": "/app/src/text_to_sql/processor.py",
  "lineno": 42,
  "function": "process_query",
  "thread": 123456,
  "process": 7890,
  "tags": {
    "component": "processor",
    "action": "complete",
    "user_id": "12345"
  },
  "data": {
    "query_id": "abc123",
    "processing_time_ms": 235,
    "tokens_used": 512
  }
}
```

## Recommended Tags

To ensure consistent logs across the system, use these standard tags:

| Tag | Description | Examples |
|-----|-------------|----------|
| `component` | System component | `db`, `llm`, `agent`, `visualization` |
| `action` | Action being performed | `start`, `complete`, `error`, `validate` |
| `agent` | Agent type when relevant | `query_understanding`, `sql_generation` |
| `db_type` | Database type | `postgres`, `mysql`, `sqlserver` |
| `llm_model` | LLM model used | `gpt-4o`, `gpt-3.5-turbo` |
| `query_id` | Unique identifier for query | UUID or hash |
| `user_id` | User identifier | User ID or session ID |
| `error_type` | Type of error | `ValueError`, `ConnectionError` |

## Integration with Monitoring Systems

### Splunk

For Splunk integration, ensure logs are in JSON format by setting:

```yaml
logging:
  json: true
  json_file: true
```

Configure Splunk forwarder to collect logs from the `logs` directory.

### Sentry

To integrate with Sentry, first install the Sentry SDK:

```bash
pip install sentry-sdk
```

Then initialize Sentry in your application:

```python
import sentry_sdk
from text_to_sql.utils.logging import configure_logging

# Initialize Sentry
sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=0.1
)

# Configure logging
configure_logging(config)
```

### ELK Stack (Elasticsearch, Logstash, Kibana)

For ELK Stack integration, configure Logstash to parse JSON logs:

```
input {
  file {
    path => "/path/to/logs/*.log"
    codec => "json"
  }
}

filter {
  # Additional processing if needed
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "text-to-sql-%{+YYYY.MM.dd}"
  }
}
```