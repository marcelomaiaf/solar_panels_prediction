FROM python:3.10-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tzdata \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src/backend

# Install Python dependencies
COPY requirements.txt ./
# Install Torch CPU explicitly, then the rest (FastAPI, Ultralytics, etc.)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn

COPY src/ .

EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

