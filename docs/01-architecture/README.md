# Architecture Documentation

This section contains comprehensive documentation about the serviceCore platform architecture, design patterns, and implementation details.

## Overview

The serviceCore platform is built with a focus on:
- **Performance**: Zig and Mojo for maximum efficiency
- **Local-First**: No external API dependencies
- **SAP BTP Integration**: HANA Cloud and Object Store
- **Modularity**: Independent, composable services

## Documents

### [Context Window Architecture](./CONTEXT_WINDOW_ARCHITECTURE.md)
Detailed explanation of the context window management system, including:
- Dynamic context allocation
- Memory management strategies
- Performance optimization
- Multi-model context handling

### [Model Orchestration Mapping](./MODEL_ORCHESTRATION_MAPPING.md)
How models are orchestrated across the platform:
- Model selection logic
- Load balancing strategies
- Fallback mechanisms
- Performance monitoring

### [RoPE Scaling Implementation](./ROPE_SCALING_IMPLEMENTATION.md)
Rotary Position Embedding (RoPE) scaling implementation:
- Mathematical foundations
- Implementation in Zig/Mojo
- Performance characteristics
- Configuration options

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│              SAP BTP Cloud                          │
│  ┌────────────────────┐  ┌────────────────────────┐│
│  │  SAP HANA Cloud    │  │  SAP Object Store      ││
│  │  (OData)           │  │  (AWS S3)              ││
│  │                    │  │                        ││
│  │ • Service Data     │  │ • Models (via DVC)     ││
│  │ • Logs/Metrics     │  │ • Documents            ││
│  │ • Traces           │  │ • Audio Files          ││
│  │ • Vectors/Graph    │  │ • Artifacts            ││
│  └────────────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────┘
                    ↑ OData/REST APIs
                    │
┌───────────────────┴─────────────────────────────────┐
│              serviceCore Platform                   │
│                                                      │
│  ┌────────────────────────────────────────────────┐│
│  │  Gateway Layer                                 ││
│  │  • nWebServe (Zig) - API Gateway              ││
│  │  • service-registry (Rust) - Discovery        ││
│  └────────────────────────────────────────────────┘│
│                                                      │
│  ┌────────────────────────────────────────────────┐│
│  │  Service Layer                                 ││
│  │  • nOpenaiServer - LLM Inference              ││
│  │  • nExtract - Document Processing             ││
│  │  • nAudioLab - Audio Processing               ││
│  │  • nCode - Code Generation                    ││
│  │  • nHyperBook, nLeanProof, nMetaData, etc.   ││
│  └────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

## Key Architectural Principles

### 1. Zero External Dependencies
- All services run locally
- No cloud API calls (OpenAI, Anthropic, etc.)
- Full data privacy and control

### 2. Performance-First
- Zig for low-level, performance-critical code
- Mojo for high-performance orchestration
- SIMD optimizations throughout

### 3. SAP BTP Native
- HANA Cloud for all structured data
- Object Store for binary data
- OData for service communication

### 4. Observability
- All logs → HANA Cloud
- All metrics → HANA Cloud
- All traces → HANA Cloud
- No third-party observability stack

### 5. Scalability
- Horizontal scaling via Docker
- Service mesh ready
- Multi-platform support (AMD64, ARM64)

## Data Flow

1. **Request Flow**:
   ```
   Client → nWebServe → service-registry → target service
   ```

2. **Logging Flow**:
   ```
   Service → HANA Cloud (OData) → Structured logs table
   ```

3. **Model Inference**:
   ```
   Service → nOpenaiServer → Local GGUF model → Response
   ```

4. **Document Processing**:
   ```
   Upload → Object Store → nExtract → Structured data → HANA Cloud
   ```

## Technology Choices

### Why Zig?
- Compile-time safety
- Zero-cost abstractions
- Excellent C interop
- Small binary sizes
- Fast compilation

### Why Mojo?
- Python-like syntax
- C/C++ performance
- SIMD by default
- Great for ML workloads
- Modern type system

### Why Rust (serviceRegistry)?
- Memory safety
- Excellent async runtime (Tokio)
- Strong ecosystem
- Production-proven

### Why SAP HANA Cloud?
- Enterprise-grade reliability
- Built-in vector engine
- Built-in graph engine
- OData support
- Integrated with SAP BTP

## Related Documentation

- [Setup Guides](../02-setup/)
- [Service Documentation](../03-services/)
- [Operations Guide](../04-operations/)

---

**Last Updated**: January 22, 2026  
**Maintained by**: Platform Architecture Team
