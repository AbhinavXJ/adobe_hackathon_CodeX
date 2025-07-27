FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . .

# Run the application
CMD ["python3", "main.py"]
