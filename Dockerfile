FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* /app/

# Configure Poetry to not use virtualenvs inside the container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-dev

# Copy application code
COPY . /app/

# Create cache directory
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Create logs directory
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Create data directory
RUN mkdir -p /app/data && chmod 777 /app/data

# Create a non-root user to run the application
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "-m", "src.text_to_sql"]