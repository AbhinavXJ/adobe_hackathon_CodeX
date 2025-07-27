FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Instruct pip to prioritize CPU wheels from the PyTorch repository
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY setup_offline.py .
RUN python3 setup_offline.py

COPY main.py .

CMD ["python3", "main.py"]
