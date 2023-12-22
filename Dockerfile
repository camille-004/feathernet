FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    make \
    clang-format \
    g++ \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt caches.

RUN pip install poetry

WORKDIR /app

COPY . .

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi
