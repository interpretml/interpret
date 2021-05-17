FROM python:3.7-slim-buster

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo bash curl git build-essential valgrind g++-multilib rsync && \
    pip install interpret

# Clean up installation excess
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove -y

# Set no entrypoint
ENTRYPOINT []
