FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# EXPLICITLY copy models directory FIRST
COPY models/ /app/models/

# Copy all other files
COPY . .

# Create directories
RUN mkdir -p /app/input /app/output

CMD ["python", "main.py"]
