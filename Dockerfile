# Python image
FROM python:3.11-slim-buster

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install --upgrade pip
RUN pip install poetry

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the source code
COPY . .

# Run Python script
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
