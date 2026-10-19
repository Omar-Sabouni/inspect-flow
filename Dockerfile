# syntax=docker/dockerfile:1

# Build the wheel in a throwaway stage so the runtime image has no build tools.
FROM python:3.12-slim AS build
WORKDIR /src
RUN python -m pip install --no-cache-dir "build>=1.2,<2"
COPY pyproject.toml README.md app.py run.py ./
COPY inspectflow ./inspectflow
RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=build /dist/*.whl /tmp/
RUN python -m pip install /tmp/*.whl && rm /tmp/*.whl \
    && useradd --create-home --uid 10001 inspectflow
USER inspectflow
WORKDIR /home/inspectflow

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=2)"]
# The container must listen on all interfaces to be reachable through a port
# mapping. Publish it on loopback only (-p 127.0.0.1:8000:8000): the service
# has no authentication.
CMD ["inspect-flow", "serve", "--host", "0.0.0.0", "--port", "8000"]
