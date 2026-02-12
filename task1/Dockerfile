
FROM python:3.11-slim

WORKDIR /app

# Minimal runtime deps (no heavy DL libs by default)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY prompts /app/prompts
COPY evaluate.py /app/evaluate.py
COPY docs /app/docs
COPY pyproject.toml /app/pyproject.toml

RUN pip install -e .

ENV PYTHONPATH=/app/src

# Default command: run evaluation (can override)
CMD ["python", "evaluate.py"]
