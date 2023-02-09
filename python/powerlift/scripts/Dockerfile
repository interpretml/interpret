# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster

COPY powerlift-0.0.1-py3-none-any.whl .

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential unixodbc-dev libpq-dev python-dev && \
    pip install powerlift-0.0.1-py3-none-any.whl[postgres,mssql,testing] && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove -y

# Set no entrypoint
ENTRYPOINT []