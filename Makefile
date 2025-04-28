# --- Load all key=value pairs from .env and export them ---
ifneq ("$(wildcard .env)","")
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif


.PHONY: setup download-data setup-db run test clean

# Configuration
PYTHON := poetry run python
CONFIG_FILE ?= config.yaml

# Default target
all: setup download-data setup-db

# Setup the project
setup:
	@echo "Setting up the project..."
	@if ! poetry env info --path >/dev/null 2>&1; then \
		echo "Virtual environment not found. Creating one..."; \
		poetry install; \
	fi
	@echo "Activating virtual environment..."
	eval $(poetry env activate)
	@echo "Environment Setup complete"

# Download sample data
download-data:
	@echo "Downloading AdventureWorks sample data..."
	$(PYTHON) src/text_to_sql/scripts/download_data.py
	@echo "Download complete"

# Setup the database
setup-db: | start-db        # order-only prerequisite
	@echo "Setting up the database..."
	$(PYTHON) src/text_to_sql/scripts/setup_db.py --config $(CONFIG_FILE)
	@echo "Database setup complete"

# Run the application
run:
	@echo "Running the application..."
	$(PYTHON) -m text_to_sql --config $(CONFIG_FILE)

# Run with standard (non-agent) approach
standard:
	@echo "Running the application with standard approach..."
	$(PYTHON) -m text_to_sql --standard --config $(CONFIG_FILE)

# Run with dynamic coordinator
dynamic:
	@echo "Running the application with dynamic coordinator..."
	$(PYTHON) -m text_to_sql --dynamic --config $(CONFIG_FILE)

# Run in debug mode
debug:
	@echo "Running the application in debug mode..."
	$(PYTHON) -m text_to_sql --debug --config $(CONFIG_FILE)

# Run with standard approach in debug mode
standard-debug:
	@echo "Running the application with standard approach in debug mode..."
	$(PYTHON) -m text_to_sql --standard --debug --config $(CONFIG_FILE)

# Run with dynamic coordinator in debug mode
dynamic-debug:
	@echo "Running the application with dynamic coordinator in debug mode..."
	$(PYTHON) -m text_to_sql --dynamic --debug --config $(CONFIG_FILE)

# Run tests
test:
	@echo "Running tests..."
	poetry run pytest
	@echo "Tests complete"

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf data/temp
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build
	@echo "Cleanup complete"

# Help information
# Help information
help:
	@echo "Available targets:"
	@echo "  setup          - Install dependencies"
	@echo "  download-data  - Download AdventureWorks sample data"
	@echo "  setup-db       - Set up the database"
	@echo "  run            - Run the application (agent-based with simple coordinator)"
	@echo "  standard       - Run the application with standard (non-agent) approach"
	@echo "  dynamic        - Run the application with dynamic coordinator"
	@echo "  debug          - Run the application in debug mode"
	@echo "  standard-debug - Run standard approach in debug mode"
	@echo "  dynamic-debug  - Run dynamic coordinator in debug mode"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean up temporary files"
	@echo "  all            - Run setup, download-data, and setup-db"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG_FILE    - Path to configuration file (default: config.yaml)"

# ---- Docker-based Postgres --------------------------------------------------
POSTGRES_CONTAINER := txt2sql-db
POSTGRES_IMAGE     := postgres:15
POSTGRES_PORT      := 5432

.PHONY: start-db stop-db

# Run a detached Postgres container if it isn’t already running
start-db:
	@if ! docker ps --format '{{.Names}}' | grep -q '^$(POSTGRES_CONTAINER)$$'; then \
		echo "Starting Postgres container …"; \
		docker run --name $(POSTGRES_CONTAINER) \
			-e POSTGRES_USER=$(TEXTTOSQL_DATABASE_USER) \
			-e POSTGRES_PASSWORD=$(TEXTTOSQL_DATABASE_PASSWORD) \
			-e POSTGRES_DB=adventureworks \
			-p $(POSTGRES_PORT):5432 \
			-v postgres_data:/var/lib/postgresql/data \
			-d $(POSTGRES_IMAGE); \
	fi
	@echo "Waiting for Postgres to accept connections …"; \
	until docker exec $(POSTGRES_CONTAINER) pg_isready -U postgres >/dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "Postgres is ready on port $(POSTGRES_PORT)"

# Tear it down when you’re done
stop-db:
	-@docker rm -f $(POSTGRES_CONTAINER) 2>/dev/null || true
	@echo "Postgres container stopped and removed"