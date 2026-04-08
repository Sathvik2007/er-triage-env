FROM python:3.11-slim

WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose port for OpenEnv
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server.app:create_app", "--host", "0.0.0.0", "--port", "7860", "--factory"]