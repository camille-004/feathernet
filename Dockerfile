FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY . .

RUN poetry install --no-interaction --no-ansi
