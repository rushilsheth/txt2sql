# Agent System Documentation

## Overview

The text-to-SQL system uses an agent-based architecture inspired by Google's A2A (architecture-to-architecture) approach. This architecture divides the process of converting natural language to SQL into distinct specialized roles, with agents that collaborate to solve complex queries.

## Agent Types

### Base Agent

All agents inherit from the `Agent` abstract base class, which provides:
- Basic logging and tracing
- Context handling
- Standardized interface

### Specialized Agents

The system includes the following specialized agents:

1. **Query Understanding Agent** 
   - Role: Extracts intent and entities from natural language
   - Input: User's query
   - Output: Intent classification and entity extraction
   - Implementation: Uses LLM to analyze the structure and meaning of the query

2. **Schema Analysis Agent**
   - Role: Identifies relevant database tables and columns
   - Input: User's query and database schema
   - Output: List of relevant tables and columns
   - Implementation: Uses LLM to match query entities to database schema elements

3. **SQL Generation Agent**
   - Role: Converts natural language to SQL
   - Input: User's query, intent, entities, and relevant schema
   - Output: SQL query
   - Implementation: Uses LLM to generate SQL based on semantic understanding

4. **Query Validation Agent**
   - Role: Validates and fixes SQL queries
   - Input: Generated SQL query
   - Output: Validated or fixed SQL query
   - Implementation: Uses database validator and LLM to fix errors

5. **Result Explanation Agent**
   - Role: Explains query results in natural language
   - Input: Query results and original query
   - Output: Natural language explanation
   - Implementation: Uses LLM to interpret and explain results

6. **Visualization Agent**
   - Role: Suggests appropriate visualizations
   - Input: Query results and original query
   - Output: Visualization configuration
   - Implementation: Uses LLM to recommend chart types and mappings

### Coordinator Agents

The system provides two types of coordinator agents:

1. **Simple Coordinator** (`SimpleCoordinatorAgent`)
   - Uses a fixed, predetermined pipeline
   - Invokes agents in a specific sequence
   - Simpler and more predictable, but less adaptable

2. **Dynamic Coordinator** (`DynamicCoordinatorAgent`)
   - Uses OpenAI function calling for decision-making
   - Decides which agent to invoke based on context
   - Includes self-reflection capability
   - More adaptable but requires OpenAI API access

## Agent Context

Agents share a common `AgentContext` object that maintains state throughout the processing pipeline. Key elements include:

- User query
- Extracted intent and entities
- Relevant schema components
- Generated SQL
- Query results
- Agent reasoning steps
- Execution metrics

## Configuration

Agent behavior can be configured through the `config.yaml` file:

```yaml
agent:
  max_iterations: 5                # Maximum iterations before stopping
  auto_fix_errors: true            # Automatically fix query errors
  use_dynamic_coordinator: false   # Use dynamic vs. simple coordinator
  
  # Per-agent settings
  query_understanding:
    use_examples: true
    max_tokens: 512
  
  # Dynamic coordinator settings
  coordinator:
    use_reflection: true
    reflection_frequency: 2
```

## Flow Diagram

The processing flow with a dynamic coordinator:

```
┌─────────────┐         ┌──────────────────┐
│ User Query  ├────────►│ Coordinator Agent│
└─────────────┘         └────────┬─────────┘
                                 │
                                 ▼
             ┌─────────────────────────────────┐
             │                                 │
     ┌───────┴───────┐               ┌────────▼─────────┐
     │ Understanding │◄──────┐       │  Schema Analysis │◄─────┐
     └───────┬───────┘       │       └────────┬─────────┘      │
             │               │                │                │
             ▼               │                ▼                │
    ┌────────────────┐       │      ┌─────────────────┐       │
    │ SQL Generation │◄──────┼──────┤ Query Validation│       │
    └────────┬───────┘       │      └────────┬────────┘       │
             │               │               │                │
             ▼               │               ▼                │
     ┌───────────────┐       │     ┌──────────────────┐      │
     │ Execute Query ├───────┘     │Result Explanation│      │
     └───────┬───────┘             └────────┬─────────┘      │
             │                              │                │
             ▼                              ▼                │
     ┌───────────────┐             ┌───────────────┐        │
     │ Visualization │◄────────────┤   Finished    ├────────┘
     └───────────────┘             └───────────────┘
```

## Adding New Agents

To add a new agent type:

1. Define a new agent class in `agent/types.py`
2. Implement the agent in `agent/llm_agents.py`
3. Register the agent with the coordinator
4. Add configuration options in `config.yaml`

Example:

```python
class CustomAgent(Agent):
    """Custom agent that does X."""
    
    def __init__(self, name: str = "Custom", config: Dict[str, Any] = None):
        super().__init__(name, AgentRole.CUSTOM, config)
    
    def process(self, context: AgentContext) -> AgentContext:
        # Implementation
        return context
```

## Debugging

The agent system provides detailed tracing and logging to help with debugging:

- Each agent logs its reasoning steps
- Execution times are recorded
- When running in debug mode, the full reasoning chain is displayed

To enable debug mode:

```bash
python -m text_to_sql --debug
# or
make debug
```