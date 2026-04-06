FROM python:3.11-slim

WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose port for OpenEnv
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "-m", "server.app"]