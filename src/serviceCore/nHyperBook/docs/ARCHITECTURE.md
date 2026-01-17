# HyperShimmy Architecture

**Version**: 1.0.0  
**Last Updated**: January 16, 2026

## Overview

HyperShimmy is a high-performance AI research assistant built with Zig (backend), Mojo (AI processing), and SAPUI5 (frontend). This document describes the system architecture, component interactions, and design decisions.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          SAPUI5 Frontend                         │
│  (FlexibleColumnLayout, Sources Panel, Chat, Audio, Slides)     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/OData V4
┌────────────────────────────┴────────────────────────────────────┐
│                      Zig OData Server                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Router    │→ │  OData Layer │→ │  Source Manager    │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Security   │  │  Error Mgmt  │  │  File Upload       │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │ FFI
┌────────────────────────────┴────────────────────────────────────┐
│                      Mojo AI Engine                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Embeddings  │  │     RAG      │  │      LLM           │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Summary   │  │   Mindmap    │  │      Slides        │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    External Services                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Qdrant    │  │   Shimmy     │  │      TTS API       │    │
│  │  (Vectors)  │  │    (LLM)     │  │    (Optional)      │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend (Zig)
- **Version**: 0.12.0+
- **Purpose**: HTTP server, OData protocol, I/O operations
- **Key Features**: Memory safety, performance, minimal dependencies

### AI Engine (Mojo)
- **Version**: Latest
- **Purpose**: AI/ML operations, embeddings, LLM inference
- **Key Features**: Python interop, high performance, GPU support

### Frontend (SAPUI5)
- **Version**: 1.120+
- **Purpose**: Enterprise UI, responsive design
- **Key Features**: OData binding, theming, accessibility

### External Services
- **Qdrant**: Vector database for embeddings
- **Shimmy**: Local LLM inference
- **TTS**: Text-to-speech (optional)

## Core Components

### 1. HTTP Server (Zig)

**Location**: `server/main.zig`

Handles all HTTP requests and routes them appropriately.

**Responsibilities**:
- Accept HTTP connections
- Route requests to handlers
- Serve static files (SAPUI5 app)
- Handle CORS
- Rate limiting
- Error responses

### 2. OData Layer (Zig)

**Location**: `server/odata*.zig`

Implements OData V4 protocol.

**Features**:
- Metadata endpoint ($metadata)
- Query options ($filter, $select, $top, $skip, $orderby, $count)
- CRUD operations
- Custom actions
- JSON serialization

### 3. Source Management (Zig)

**Location**: `server/sources.zig`

Manages research sources and their lifecycle.

**Operations**:
- Create, Read, Update, Delete sources
- Status tracking (Pending → Processing → Ready/Error)
- Content extraction
- Metadata management

### 4. File Upload (Zig)

**Location**: `server/upload.zig`

Handles multipart file uploads.

**Features**:
- Multipart form data parsing
- File validation (type, size)
- Filename sanitization
- Temporary storage
- Integration with source management

### 5. I/O Subsystem (Zig)

**Location**: `io/*.zig`

Handles external data fetching.

**Modules**:
- `http_client.zig` - HTTP requests
- `html_parser.zig` - HTML parsing
- `pdf_parser.zig` - PDF text extraction
- `web_scraper.zig` - Web content extraction

### 6. Embedding Engine (Mojo)

**Location**: `mojo/embeddings.mojo`

Generates semantic embeddings for text.

**Features**:
- Text vectorization
- Batch processing
- Normalization
- Caching

### 7. RAG Pipeline (Mojo)

**Location**: `mojo/rag.mojo`

Retrieval Augmented Generation for chat.

**Process**:
1. Query embedding generation
2. Vector similarity search
3. Context retrieval
4. LLM prompt construction
5. Response generation

### 8. Content Generators (Mojo)

**Summary**: `mojo/summary.mojo`  
**Mindmap**: `mojo/mindmap.mojo`  
**Slides**: `mojo/slide_generator.mojo`

Generate various content types from sources.

### 9. Security Module (Zig)

**Location**: `server/security.zig`

Security features and validation.

**Features**:
- Input validation
- Sanitization (HTML, SQL, path)
- Rate limiting
- CORS management
- CSP headers
- Token generation

### 10. Error Handling (Zig)

**Location**: `server/errors.zig`

Centralized error management.

**Features**:
- Error categorization
- OData error formatting
- HTTP status mapping
- Error logging
- Recovery strategies

## Data Flow

### Document Ingestion Flow

```
1. User uploads PDF/URL
   ↓
2. Zig validates & saves file
   ↓
3. Creates Source (status: Pending)
   ↓
4. Zig extracts text (PDF parser/web scraper)
   ↓
5. Source status → Processing
   ↓
6. Mojo generates embeddings
   ↓
7. Store vectors in Qdrant
   ↓
8. Source status → Ready
```

### Chat/RAG Flow

```
1. User sends query
   ↓
2. Mojo generates query embedding
   ↓
3. Search Qdrant for similar vectors
   ↓
4. Retrieve relevant source chunks
   ↓
5. Build LLM prompt with context
   ↓
6. Shimmy generates response
   ↓
7. Stream response to user
```

### Audio Generation Flow

```
1. User selects sources
   ↓
2. Mojo generates summary text
   ↓
3. Send to TTS API
   ↓
4. Receive audio file
   ↓
5. Save & return URL to user
```

## Design Patterns

### 1. Repository Pattern
Source management uses repository pattern for data access abstraction.

### 2. Strategy Pattern
Different source types (PDF, URL, Text) use strategy pattern for processing.

### 3. Factory Pattern
Error creation uses factory pattern for consistent error objects.

### 4. Observer Pattern
Status changes notify interested components.

### 5. Pipeline Pattern
Document processing flows through a pipeline of transformations.

## Scalability Considerations

### Horizontal Scaling
- Stateless server design
- Session data in external store
- Load balancer compatible

### Vertical Scaling
- Efficient memory usage
- Minimal allocations
- Zero-copy where possible

### Caching Strategy
- Embedding cache in memory
- LLM response cache
- Vector search results cache

### Database Strategy
- SQLite for metadata (development)
- PostgreSQL for production
- Qdrant for vectors

## Security Architecture

### Input Validation
All user inputs validated before processing.

### Sanitization
HTML, SQL, and path sanitization applied.

### Rate Limiting
Per-client rate limits prevent abuse.

### CORS
Configurable origins for frontend access.

### CSP
Content Security Policy headers on responses.

## Performance Optimizations

### Zig Optimizations
- Compile-time execution
- Inline functions
- Manual memory management
- Avoid allocations in hot paths

### Mojo Optimizations
- SIMD vectorization
- GPU acceleration (when available)
- Parallel processing
- Efficient tensor operations

### Network Optimizations
- HTTP keep-alive
- Compression
- Chunked transfer encoding
- Connection pooling

## Monitoring & Observability

### Metrics
- Request counts
- Response times
- Error rates
- Queue depths
- Memory usage

### Logging
- Structured logging
- Log levels (DEBUG, INFO, WARN, ERROR)
- Request/response logging
- Error traces

### Health Checks
- `/health` endpoint
- Dependency status
- Resource usage

## Testing Strategy

### Unit Tests
- Individual function testing
- Mock dependencies
- Edge case coverage

### Integration Tests
- Component interaction testing
- Workflow validation
- End-to-end scenarios

### Performance Tests
- Load testing
- Stress testing
- Endurance testing

## Deployment Architecture

### Development
- Local Zig server
- Local Mojo runtime
- Docker containers for services

### Production
- Containerized deployment
- Kubernetes orchestration
- Service mesh
- CDN for static assets

## Future Enhancements

1. **Authentication**: OAuth2/OIDC integration
2. **Multi-tenancy**: Isolated user workspaces
3. **Streaming**: WebSocket support
4. **Offline**: Progressive web app
5. **Analytics**: Usage tracking
6. **Collaboration**: Real-time sharing

---

**For detailed API reference**, see [API.md](API.md)  
**For development guide**, see [DEVELOPER.md](DEVELOPER.md)  
**For deployment guide**, see [DEPLOYMENT.md](DEPLOYMENT.md)
