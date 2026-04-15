# Changelog

All notable changes to Project Prometheus will be documented in this file.

## [0.1.0] - 2026-04-15

### 🎉 Initial Public Release

**Core Framework**
- Chat loop with streaming and non-streaming LLM support
- 40+ tool handlers with automatic dispatch and middleware
- Tool pipeline execution (sequential and parallel)
- Circuit breaker pattern for tool resilience

**Memory System**
- 5-layer retrieval: preferences → entities → semantic → fuzzy → episodes
- Rust-powered hybrid search (SIMD cosine + BM25 + trigram fuzzy)
- Automatic memory consolidation and deduplication
- Entity extraction with relation classification
- Token-budgeted context injection
- **95.7% R@5 on LongMemEval** — state-of-the-art

**Identity System**
- Self-evolving AI personality (SOUL.md, IDENTITY.md)
- User modeling (USER.md)
- Belief crystallization (BELIEFS.md)
- Reflection cycles and journaling
- Heartbeat-driven background processing

**LLM Providers**
- Anthropic (Claude), OpenAI (GPT-4), Google (Gemini)
- OpenRouter (100+ models)
- In-house SDK — zero vendor dependencies
- Automatic failover and circuit breakers

**Tools**
- Code execution (Python, Node, Go, Rust, Java, C++, Ruby, PHP, R, Julia, Kotlin, TypeScript)
- macOS desktop automation (PyObjC native)
- Chrome browser control (CDP protocol)
- Android device control (ADB)
- File system operations
- Git operations with semantic code search
- Web search and deep research
- Memory store and recall
- MCTS strategic planning
- Multi-agent orchestration
- Voice notes and TTS
- Image generation
- PDF reading
- Diagram generation (Mermaid)
- Scheduled tasks (cron)

**Channels**
- CLI with rich terminal UI
- Telegram bot
- Discord bot
- WhatsApp (via Twilio)
- Webchat widget

**Security**
- Ethical Governance Module (EGM)
- JWT authentication
- API key management
- Secret auto-redaction
- Docker sandboxed execution
- Trust levels (YOLO / Normal / Paranoid)

**Infrastructure**
- Docker Compose deployment
- Kubernetes manifests
- Grafana dashboards
- Prometheus metrics

**Benchmarks**
- LongMemEval: 0.9574 R@5 (22 points ahead of SOTA)
- Full evaluation harness for ALE-bench, SWE-bench
