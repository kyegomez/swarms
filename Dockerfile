# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-alpine AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN uv venv /opt/venv --python /usr/local/bin/python

WORKDIR /src
COPY pyproject.toml README.md ./
COPY swarms ./swarms
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python .

# Same base as the builder, so the venv's interpreter path still resolves.
FROM python:${PYTHON_VERSION}-alpine

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The base image's pip is never used and bundles outdated urllib3, setuptools and msgpack.
RUN apk upgrade --no-cache \
    && apk add --no-cache bash \
    && rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/bin/pip* \
    && adduser -D -u 1000 -s /bin/bash swarms \
    && mkdir /app \
    && chown swarms:swarms /app

COPY --from=builder /opt/venv /opt/venv

USER swarms
WORKDIR /app

CMD ["python"]
