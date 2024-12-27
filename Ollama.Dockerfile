# Stage 1: Temporary container to download the model
FROM ollama/ollama:latest as downloader

# Start the Ollama server in the background
RUN nohup ollama serve & \
    sleep 5 && \
    ollama pull llama3.1

# Stage 2: Final container with preloaded model
FROM ollama/ollama:latest

# Install bash shell (if needed)
RUN apt-get update && apt-get install -y bash

# Copy the model from the first stage
COPY --from=downloader /root/.ollama/models/ /root/.ollama/models/

# Ensure the PATH is correctly set
ENV PATH="/usr/local/bin:$PATH"

# Expose the API port (default is 11434)
EXPOSE 11434

# Keep the container alive indefinitely using bash
CMD ["/bin/bash", "-c", "tail -f /dev/null"]