FROM python:3.12-slim

# Install system deps (curl for Poetry installer)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Claude Code refuses --dangerously-skip-permissions as root)
RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser

# Install Poetry for appuser
USER appuser
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Note: Claude Code CLI is bundled with claude-agent-sdk >= 0.1.8
# No separate Node.js/npm installation required

# Copy the app code
COPY --chown=appuser:appuser . /app

# Set working directory
WORKDIR /app

# Install Python dependencies with Poetry
RUN poetry install --no-root

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn (development mode with reload; switch to --no-reload for prod)
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]