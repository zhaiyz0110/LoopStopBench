# CPU harness for collection, replay, analysis, and restricted code evaluation.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install a self-contained copy so the restricted code-exec service needs no
# host repository mount. The networked harness may still mount a development
# checkout at /workspace.
COPY pyproject.toml README.md /opt/loopstop-src/
COPY src /opt/loopstop-src/src
RUN pip install --no-cache-dir "/opt/loopstop-src[full,dev]"

COPY scripts /opt/loopstop/scripts
COPY configs /opt/loopstop/configs

RUN useradd -m runner \
    && mkdir -p /workspace /work \
    && chown runner:runner /workspace /work

USER runner
WORKDIR /workspace

CMD ["bash"]

