# predacore/sandbox — unified multi-language sandbox image.
#
# The `execute_code` tool and the Docker sandbox manager pull this image
# once and reuse it for every sandboxed run. Ships with python, node, go,
# ruby, rust, java, php, r, julia, typescript preinstalled so the user
# never waits for a one-off `docker pull python:3.12` mid-conversation.
#
# Built and pushed to Docker Hub by .github/workflows/docker-sandbox.yml
# on release tags. Users pull via `predacore bootstrap` (which calls
# `docker pull predacore/sandbox:latest` when Docker is present).
#
# Base: Debian slim — small footprint, friendly apt toolchain, stable
# cross-arch support (linux/amd64 + linux/arm64 via buildx).

ARG DEBIAN_VERSION=bookworm
FROM debian:${DEBIAN_VERSION}-slim

LABEL org.opencontainers.image.title="predacore/sandbox"
LABEL org.opencontainers.image.description="Multi-language sandbox runtime for PredaCore's execute_code tool"
LABEL org.opencontainers.image.source="https://github.com/rockshub1/predacore"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_ENV=production \
    PATH=/usr/local/cargo/bin:/usr/local/go/bin:$PATH \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup

# ── System foundation ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      # Build toolchain (needed by compiled runtimes + some pip packages)
      build-essential \
      pkg-config \
      ca-certificates \
      curl \
      wget \
      git \
      jq \
      unzip \
      xz-utils \
    # Python 3.11 (default on Debian bookworm)
      python3 python3-pip python3-venv python3-dev \
    # Node 20 (via NodeSource) — see below
    # Ruby
      ruby-full \
    # PHP
      php-cli \
    # R
      r-base \
    # Julia (via distro — quick)
      julia \
    # Java 17
      openjdk-17-jdk-headless \
  && rm -rf /var/lib/apt/lists/*

# ── Node.js 20 LTS + TypeScript + common tools ────────────────────────
# M60 (Wave 7) note: `curl | bash` from NodeSource pulls a setup script
# whose contents change without versioning — supply-chain trust is
# entirely on NodeSource's TLS. To pin tighter:
#   1. Download a specific NodeSource setup commit by its sha:
#      curl -fsSL https://raw.githubusercontent.com/nodesource/distributions/<commit>/scripts/setup_20.x | bash -
#   2. Or skip NodeSource and install Node from the official tarball
#      with a pinned sha256 (mirrors the Go install pattern below).
# Kept as-is for now because the alternative is significantly more code
# and NodeSource is the documented standard.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && npm install -g npm@latest typescript ts-node \
  && rm -rf /var/lib/apt/lists/*

# M59 (Wave 7) follow-up: npm@latest, typescript, ts-node above are
# unpinned — a fresh build pulls today's latest of each. To pin, replace
# with explicit versions like:
#   npm install -g npm@10.8.2 typescript@5.6.2 ts-node@10.9.2
# Same applies to the Python `pip install` and `rustup` blocks below.
# Tracked as M59 follow-up; pinning is a separate cleanup pass that
# needs reproducibility validation (npm-check-updates or Renovate).

# ── Go 1.22 ───────────────────────────────────────────────────────────
ARG GO_VERSION=1.22.5
RUN ARCH="$(dpkg --print-architecture)" && \
    case "$ARCH" in \
      amd64) GOARCH="amd64" ;; \
      arm64) GOARCH="arm64" ;; \
      *) echo "unsupported arch $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${GOARCH}.tar.gz" -o /tmp/go.tgz && \
    tar -C /usr/local -xzf /tmp/go.tgz && \
    rm /tmp/go.tgz

# ── Rust (stable, minimal profile) ────────────────────────────────────
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- --default-toolchain stable --profile minimal -y && \
    chmod -R a+rX /usr/local/cargo /usr/local/rustup

# ── Python baseline packages (pre-warm the common ones) ───────────────
RUN python3 -m pip install --break-system-packages --upgrade pip && \
    python3 -m pip install --break-system-packages \
      numpy pandas requests httpx pydantic rich matplotlib ipython

# ── Unprivileged user — code runs as this, not root ───────────────────
RUN useradd --create-home --shell /bin/bash --uid 1000 sandbox && \
    mkdir -p /workspace && chown sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

# Default: run the user-provided script. The sandbox manager mounts
# /workspace read-write and passes the entrypoint at runtime.
CMD ["bash"]
