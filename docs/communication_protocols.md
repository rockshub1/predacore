# PredaCore: Inter-Component Communication Protocols (v1)

**Version:** 0.1
**Date:** 2025-04-10

This document outlines the initial decisions regarding communication protocols between the core internal components of PredaCore and for external interactions.

---

## 1. Internal Core Component Communication (Synchronous)

**Chosen Protocol:** **gRPC**

**Rationale:**

*   **Performance:** gRPC generally offers lower latency and higher throughput compared to REST/JSON due to its use of HTTP/2 and Protocol Buffers. This is crucial for the potentially high volume of interactions between core components (CSC, DAF, KN, EGM, UME, WIL).
*   **Efficient Serialization:** Protocol Buffers provide a compact binary format, reducing network bandwidth usage and serialization/deserialization overhead.
*   **Strong Typing & Schema Definition:** Defining service interfaces and message types using `.proto` files enforces contracts between components, improving maintainability and reducing integration errors.
*   **Streaming Support:** Native support for bidirectional streaming might be valuable for future use cases (e.g., continuous data feeds, long-running agent interactions).

**Implementation Notes:**

*   `.proto` files defining services and messages will be maintained, likely within a shared `src/common/protos` directory or similar.
*   Initial focus will be on unary (request/response) RPC calls. Streaming will be adopted as needed.
*   Libraries like `grpcio` and `grpcio-tools` will be added as dependencies in `pyproject.toml`.

## 2. Asynchronous Communication / Eventing (Future Consideration)

**Potential Protocol:** **Message Queue (e.g., RabbitMQ, Kafka, NATS)**

**Rationale:**

*   **Decoupling:** Allows components to operate independently without direct knowledge of consumers.
*   **Asynchronous Processing:** Suitable for tasks that don't require immediate response (e.g., background knowledge ingestion, logging, triggering workflows).
*   **Resilience:** Can buffer messages if a consumer is temporarily unavailable.

**Implementation Notes:**

*   This will likely be introduced in later phases as specific needs for asynchronous processing or event-driven architectures arise. The choice of broker will depend on specific requirements (e.g., persistence, ordering guarantees, throughput).

## 3. External API (User/Developer Facing)

**Chosen Protocol:** **REST (likely via FastAPI)**

**Rationale:**

*   **Simplicity & Ubiquity:** RESTful APIs over HTTP/JSON are widely understood and easily consumable by various clients (web browsers, other applications).
*   **Standardization:** Leverages standard HTTP methods, status codes, and conventions.
*   **Ecosystem:** Excellent tooling and library support (e.g., FastAPI in Python provides automatic validation, documentation via OpenAPI).

**Implementation Notes:**

*   The API component (`src/api/`) will implement the REST endpoints defined in Section 8 of the design document.
*   Data validation (e.g., using Pydantic) will be crucial.
*   Authentication and authorization mechanisms (e.g., OAuth2, API Keys) must be implemented.

---

**Summary:**

*   **Internal Sync:** gRPC
*   **Internal Async:** Message Queue (Future)
*   **External:** REST