FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first before anything else
RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 'dashboard.app:app'
