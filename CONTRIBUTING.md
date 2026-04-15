# Contributing to Project Prometheus

Thank you for your interest in contributing! Prometheus is an open-source
AI agent framework, and we welcome contributions of all kinds.

## Getting Started

### Prerequisites

- Python 3.10+
- Rust toolchain (for the `jarvis_core` crate)
- macOS (primary platform; Linux support is in progress)

### Setup

```bash
# Clone the repo
git clone https://github.com/anthropic-ai/project-prometheus.git
cd project-prometheus

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Build the Rust crate
cd src/jarvis_core_crate
maturin develop --release
cd ../..

# Run tests
pytest
```

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](../../issues/new?template=bug_report.md) template
- Include your OS version, Python version, and steps to reproduce
- Attach relevant logs (redact any API keys!)

### Suggesting Features

- Use the [Feature Request](../../issues/new?template=feature_request.md) template
- Describe the use case, not just the solution
- Check existing issues first

### Submitting Code

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/my-feature`
3. **Write tests** for new functionality
4. **Run the test suite**: `pytest`
5. **Run linting**: `ruff check src/`
6. **Submit a PR** with a clear description of what changed and why

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length: 88 characters
- Type hints are encouraged but not strictly enforced (yet)
- Docstrings for public functions and classes

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new browser automation tool
fix: resolve memory leak in vector search
docs: update installation guide
test: add coverage for entity extraction
refactor: simplify LLM provider router
```

## Architecture Overview

```
src/jarvis/
├── agents/          # Multi-agent orchestration, DAF bridge
├── auth/            # JWT, API keys, sandbox security
├── channels/        # Telegram, Discord, WhatsApp, webchat
├── identity/        # Self-evolving AI identity system
├── llm_providers/   # Anthropic, OpenAI, Gemini, router
├── memory/          # Store, retriever, consolidator
├── operators/       # Desktop, browser, Android, screen vision
├── services/        # Cron, DB, embedding, git, voice
├── tools/           # 40+ tool handlers, dispatcher, middleware
├── _vendor/         # CSC, DAF, EGM, KN, UME microservices
├── evals/           # LongMemEval, ALE-bench, SWE-bench
├── cli.py           # Main CLI entry point
├── core.py          # Core chat loop
└── config.py        # Configuration management
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/jarvis

# Run specific test module
pytest tests/jarvis/test_memory.py

# Run tests that hit real APIs (requires credentials)
pytest -m real
```

## Code of Conduct

Be kind. Be constructive. We're building something cool together.

## License

By contributing, you agree that your contributions will be licensed
under the Apache License 2.0.
