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
#
# Pinning posture (H30/M59/M60, Wave 12 — 2026-05-11):
#   - All npm + Python + Rust + Go versions are pinned to specific
#     semver. A clean rebuild produces the same versions deterministically.
#   - Debian base image is pinned by version codename (bookworm) but NOT
#     by sha256 digest. To go full-digest-pin, set BASE_IMAGE_DIGEST
#     below to the output of:
#       docker pull debian:bookworm-slim
#       docker inspect debian:bookworm-slim --format='{{index .RepoDigests 0}}'
#     then change `FROM ${BASE_IMAGE}:${DEBIAN_VERSION}-slim` to
#     `FROM debian@${BASE_IMAGE_DIGEST}`.
#   - The Node tarball IS pinned by sha256 (replaces the prior
#     NodeSource `curl | bash` supply-chain hop).
#   - Bumping anything: change the ARG below, rebuild, run
#     `scripts/regen-sandbox-shas.sh` (TODO) if SHA inputs are altered.

ARG DEBIAN_VERSION=bookworm
# OPTIONAL: digest-pin the base image. Leave empty to use the tag.
# To pin: ARG BASE_IMAGE_DIGEST=sha256:abc123... (see header).
ARG BASE_IMAGE_DIGEST=

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
    PATH=/usr/local/node/bin:/usr/local/cargo/bin:/usr/local/go/bin:$PATH \
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
    # Node 20 — installed from official tarball below
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

# ── Node.js 20 LTS via OFFICIAL tarball (sha256-pinned) ───────────────
# Replaces the prior `curl https://deb.nodesource.com/setup_20.x | bash`
# pattern (M60 fix). Trust chain is now: nodejs.org TLS + sha256 of the
# tarball we explicitly verify. No third-party setup script.
#
# Update procedure:
#   1. Pick a new version from https://nodejs.org/en/download/releases/
#   2. From https://nodejs.org/dist/v<VERSION>/SHASUMS256.txt grab the
#      sha256 for node-v<VERSION>-linux-x64.tar.xz   (NODE_SHA256_AMD64)
#      and  node-v<VERSION>-linux-arm64.tar.xz       (NODE_SHA256_ARM64)
#   3. Bump NODE_VERSION + both shas below.
ARG NODE_VERSION=20.18.0
ARG NODE_SHA256_AMD64=4543670b589593f8fa5f106111fd5139081da42bb165a9239f05195e405f240a
ARG NODE_SHA256_ARM64=a9ce85675ba33f00527f6234d90000946c0936fb4fca605f1891bb5f4fe6fb0a
ARG NPM_VERSION=10.8.3
ARG TYPESCRIPT_VERSION=5.6.3
ARG TSNODE_VERSION=10.9.2

RUN ARCH="$(dpkg --print-architecture)" && \
    case "$ARCH" in \
      amd64) NODE_ARCH="x64"; NODE_SHA="$NODE_SHA256_AMD64" ;; \
      arm64) NODE_ARCH="arm64"; NODE_SHA="$NODE_SHA256_ARM64" ;; \
      *) echo "unsupported arch $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" -o /tmp/node.tar.xz && \
    echo "${NODE_SHA}  /tmp/node.tar.xz" | sha256sum -c - && \
    mkdir -p /usr/local/node && \
    tar -C /usr/local/node --strip-components=1 -xJf /tmp/node.tar.xz && \
    rm /tmp/node.tar.xz && \
    npm install -g npm@${NPM_VERSION} typescript@${TYPESCRIPT_VERSION} ts-node@${TSNODE_VERSION}

# ── Go 1.22 (tarball — already pinned via GO_VERSION) ─────────────────
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

# ── Rust (pinned stable toolchain, minimal profile) ───────────────────
# Pinned to a specific stable release rather than `stable` channel so a
# rebuild months later doesn't silently jump major versions. Bump
# RUST_VERSION when you want the upgrade.
ARG RUST_VERSION=1.83.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- --default-toolchain ${RUST_VERSION} --profile minimal -y && \
    chmod -R a+rX /usr/local/cargo /usr/local/rustup

# ── Python baseline packages (pinned) ─────────────────────────────────
# All versions pinned to a known-compatible set. The `pip install pip`
# pin avoids surprise behavior changes from pip itself.
RUN python3 -m pip install --break-system-packages --upgrade 'pip==24.3.1' && \
    python3 -m pip install --break-system-packages \
      'numpy==2.1.3' \
      'pandas==2.2.3' \
      'requests==2.32.3' \
      'httpx==0.28.1' \
      'pydantic==2.10.3' \
      'rich==13.9.4' \
      'matplotlib==3.10.0' \
      'ipython==8.30.0'

# ── Unprivileged user — code runs as this, not root ───────────────────
RUN useradd --create-home --shell /bin/bash --uid 1000 sandbox && \
    mkdir -p /workspace && chown sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

# Default: run the user-provided script. The sandbox manager mounts
# /workspace read-write and passes the entrypoint at runtime.
CMD ["bash"]
