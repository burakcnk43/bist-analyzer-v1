FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure start script is executable
RUN chmod +x start.sh

# Expose port (Railway will provide this via env)
EXPOSE 8000

# Use the PORT env variable correctly
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
