#!/bin/bash
chmod +x start.sh
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
