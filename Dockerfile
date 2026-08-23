FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY nsagent /app/nsagent
COPY nsagent_api /app/nsagent_api
COPY sample_project /app/sample_project
COPY agent_data /app/agent_data

ENV PYTHONPATH=/app
ENV AGENT_PROJECT_ROOT=/app/sample_project
ENV AGENT_STATE_PATH=/app/agent_state/final_model_v3.json
ENV PORT=10000

EXPOSE 10000

CMD ["sh", "-c", "uvicorn nsagent_api.main:app --host 0.0.0.0 --port $PORT"]
