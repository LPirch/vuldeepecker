FROM python:3.12-slim

RUN apt update && \
    apt install -y git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /vuldeepecker

WORKDIR /vuldeepecker

RUN pip install --no-cache-dir -r requirements.txt
