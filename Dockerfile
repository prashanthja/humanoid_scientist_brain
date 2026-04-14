FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["sh", "-c", "python -m gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --timeout 120 dashboard.app:app"]
