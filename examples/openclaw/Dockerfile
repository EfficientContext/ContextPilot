FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir contextpilot

EXPOSE 8765

ENTRYPOINT ["python", "-m", "contextpilot.server.http_server", "--stateless"]
CMD ["--port", "8765", "--infer-api-url", "https://api.anthropic.com"]
