FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port (Railway will provide this via env)
EXPOSE 8000

CMD ["./start.sh"]
