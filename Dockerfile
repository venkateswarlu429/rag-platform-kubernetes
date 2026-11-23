# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Non-root: the image runs as an unprivileged user.
RUN useradd --create-home --uid 10001 rag && chown -R rag:rag /app
USER 10001

EXPOSE 8000

# Defaults are overridable at deploy time via the ConfigMap / env vars.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
