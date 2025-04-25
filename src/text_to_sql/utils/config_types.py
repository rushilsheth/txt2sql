from __future__ import annotations

"""Refactored configuration module.

A lightweight declarative approach that removes boilerplate by:
- factoring common `from_dict` / `to_dict` behavior into a single `BaseConfig`
- relying on `dataclasses.asdict` for serialization
- keeping each section focused only on its own defaults or helper methods

"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Common helpers                                                               
# ---------------------------------------------------------------------------

@dataclass
class BaseConfig:
    """Extend for typed configuration sections."""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None = None):  # type: ignore[valid-type]
        raw = raw or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__annotations__})  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Leaf‑level sections                                                          
# ---------------------------------------------------------------------------

@dataclass
class DatabaseConfig(BaseConfig):
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

    # Helper untouched – still convenient for DB libraries
    def get_connection_params(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "sslmode": self.ssl_mode,
        }


@dataclass
class LLMConfig(BaseConfig):
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout: int = 30
    max_tokens: int = 1_024
    use_cache: bool = True
    cache_ttl: int = 3_600  # seconds


@dataclass
class LoggingConfig(BaseConfig):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    directory: str = "logs"
    colored: bool = True
    structured: bool = True
    json: bool = False
    json_file: bool = False
    component_levels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentConfig(BaseConfig):
    max_iterations: int = 5
    auto_fix_errors: bool = True
    use_dynamic_coordinator: bool = False

    # Sub‑agents can accept arbitrary key/value overrides
    query_understanding: Dict[str, Any] = field(default_factory=dict)
    schema_analysis: Dict[str, Any] = field(default_factory=dict)
    sql_generation: Dict[str, Any] = field(default_factory=dict)
    query_validation: Dict[str, Any] = field(default_factory=dict)
    result_explanation: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)
    coordinator: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig(BaseConfig):
    title: str = "Text‑to‑SQL Interface"
    description: str = "Ask questions about your data in natural language"
    theme: str = "default"
    debug_mode: bool = False
    use_semantic_engine: bool = True
    host: str = "0.0.0.0"
    port: int = 7_860
    share: bool = False


# ---------------------------------------------------------------------------
# Root section                                                                 
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig(BaseConfig):
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    app: AppConfig = field(default_factory=AppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]):  # type: ignore[override]
        return cls(
            database=DatabaseConfig.from_dict(raw.get("database")),
            llm=LLMConfig.from_dict(raw.get("llm")),
            app=AppConfig.from_dict(raw.get("app")),
            logging=LoggingConfig.from_dict(raw.get("logging")),
            agent=AgentConfig.from_dict(raw.get("agent")),
        )
