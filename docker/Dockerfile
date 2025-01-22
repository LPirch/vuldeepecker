FROM python:3.12-slim

RUN apt update && \
    apt install -y git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /vuldeepecker

WORKDIR /vuldeepecker

SHELL ["/bin/bash", "-c"]

RUN python -m venv .venv && \
    source .venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt
