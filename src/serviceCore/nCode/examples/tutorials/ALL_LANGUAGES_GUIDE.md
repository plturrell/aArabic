# nCode Multi-Language Indexing Guide

Comprehensive guide for indexing projects in all supported languages with nCode.

## Table of Contents

1. [Python](#python)
2. [Java](#java)
3. [Rust](#rust)
4. [Go](#go)
5. [Data Languages](#data-languages)
6. [Other Languages](#other-languages)

---

## Python

### Installation

```bash
pip install scip-python
```

### Basic Usage

```bash
# Index Python project
scip-python index .

# With specific output
scip-python index . --output index.scip

# Load to nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### Example Project Structure

```python
# src/models/user.py
class User:
    def __init__(self, id: str, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
    
    def validate_email(self) -> bool:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, self.email))

# src/services/auth.py
from models.user import User

class AuthService:
    def authenticate(self, email: str, password: str) -> User | None:
        """Authenticate user with credentials"""
        # Implementation here
        pass
```

### Virtual Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
scip-python index .
```

### Export to Memgraph

```bash
python scripts/load_to_databases.py index.scip \
  --memgraph \
  --memgraph-host localhost \
  --memgraph-port 7687
```

### Query with Cypher

```cypher
// Find all classes in the project
MATCH (c:Symbol {kind: 'class'})
RETURN c.name, c.file
LIMIT 10

// Find all methods in a specific class
MATCH (c:Symbol {name: 'AuthService', kind: 'class'})
      -[:ENCLOSES]->(m:Symbol {kind: 'method'})
RETURN m.name, m.documentation

// Find inheritance relationships
MATCH (child:Symbol)-[:IMPLEMENTS]->(parent:Symbol)
RETURN child.name, parent.name
```

---

## Java

### Installation

Download from [scip-java releases](https://github.com/sourcegraph/scip-java/releases)

```bash
# Using coursier (recommended)
cs install scip-java

# Or download binary
wget https://github.com/sourcegraph/scip-java/releases/download/v0.9.8/scip-java
chmod +x scip-java
```

### Maven Project

```bash
# Add to pom.xml
<build>
  <plugins>
    <plugin>
      <groupId>com.sourcegraph</groupId>
      <artifactId>scip-java</artifactId>
      <version>0.9.8</version>
    </plugin>
  </plugins>
</build>

# Generate index
mvn clean compile
scip-java index
```

### Gradle Project

```groovy
// build.gradle
plugins {
    id 'java'
    id 'com.sourcegraph.scip-java' version '0.9.8'
}

// Generate index
./gradlew scipJava
```

### Example Code

```java
// src/main/java/com/example/model/User.java
package com.example.model;

public class User {
    private String id;
    private String name;
    private String email;
    
    public User(String id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }
    
    public boolean validateEmail() {
        return email.matches("^[A-Za-z0-9+_.-]+@(.+)$");
    }
}

// src/main/java/com/example/service/AuthService.java
package com.example.service;

import com.example.model.User;

public class AuthService {
    public User authenticate(String email, String password) {
        // Implementation
        return null;
    }
}
```

---

## Rust

### Installation

```bash
rustup component add rust-analyzer
cargo install rust-analyzer
```

### Basic Usage

```bash
# Ensure Cargo.toml exists
cargo build

# Generate SCIP index using rust-analyzer
rust-analyzer scip .

# Load to nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### Example Code

```rust
// src/models/user.rs
pub struct User {
    pub id: String,
    pub name: String,
    pub email: String,
}

impl User {
    pub fn new(id: String, name: String, email: String) -> Self {
        User { id, name, email }
    }
    
    pub fn validate_email(&self) -> bool {
        self.email.contains('@')
    }
}

// src/services/auth.rs
use crate::models::user::User;

pub struct AuthService {
    users: Vec<User>,
}

impl AuthService {
    pub fn authenticate(&self, email: &str, password: &str) -> Option<&User> {
        self.users.iter().find(|u| u.email == email)
    }
}
```

### Cargo.toml

```toml
[package]
name = "my_rust_project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
```

---

## Go

### Installation

```bash
go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
```

### Basic Usage

```bash
# Initialize Go module
go mod init myproject

# Generate index
scip-go

# Load to nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### Example Code

```go
// models/user.go
package models

import "regexp"

type User struct {
    ID    string
    Name  string
    Email string
}

func NewUser(id, name, email string) *User {
    return &User{
        ID:    id,
        Name:  name,
        Email: email,
    }
}

func (u *User) ValidateEmail() bool {
    emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
    return emailRegex.MatchString(u.Email)
}

// services/auth.go
package services

import "myproject/models"

type AuthService struct {
    users map[string]*models.User
}

func NewAuthService() *AuthService {
    return &AuthService{
        users: make(map[string]*models.User),
    }
}

func (s *AuthService) Authenticate(email, password string) (*models.User, error) {
    user, exists := s.users[email]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }
    return user, nil
}
```

---

## Data Languages

### Supported Formats

- JSON
- XML
- YAML
- TOML
- SQL
- GraphQL
- Protocol Buffers
- Markdown
- HTML/CSS

### Using nCode Tree-Sitter Indexer

```bash
# Index JSON files
./zig-out/bin/ncode-treesitter index \
  --language json \
  --output index.scip \
  ./data

# Index YAML configs
./zig-out/bin/ncode-treesitter index \
  --language yaml \
  --output index.scip \
  ./config

# Index SQL schemas
./zig-out/bin/ncode-treesitter index \
  --language sql \
  --output schema.scip \
  ./database

# Load to nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### Example Files

```json
// data/config.json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "mydb"
  },
  "api": {
    "baseUrl": "https://api.example.com",
    "timeout": 30000
  }
}
```

```yaml
# config/app.yaml
server:
  port: 8080
  host: localhost

database:
  driver: postgres
  connection: postgresql://localhost:5432/mydb

features:
  - authentication
  - authorization
  - logging
```

```sql
-- database/schema.sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
```

---

## Other Languages

### C#

```bash
# Install scip-dotnet
dotnet tool install -g scip-dotnet

# Index project
scip-dotnet index

# Load to nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### Ruby

```bash
# Install scip-ruby
gem install scip-ruby

# Index project
scip-ruby index

# Load to nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### Kotlin

```bash
# Using scip-java (supports Kotlin)
scip-java index --build-tool gradle

# Load to nCode
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

---

## Universal Workflow

Regardless of language, the workflow is consistent:

### 1. Install Language Indexer

```bash
# Find your language's indexer
# TypeScript: npm install -g @sourcegraph/scip-typescript
# Python: pip install scip-python
# Java: cs install scip-java
# Rust: rustup component add rust-analyzer
# Go: go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
```

### 2. Generate SCIP Index

```bash
# Each indexer has similar commands
<indexer> index [options]
# Outputs: index.scip
```

### 3. Load to nCode

```bash
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### 4. Query Code Intelligence

```bash
# Find definitions
curl -X POST http://localhost:18003/v1/definition \
  -H "Content-Type: application/json" \
  -d '{"file": "src/main.ext", "line": 10, "character": 5}'

# Find references
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{"symbol": "package.Class#method()."}'
```

### 5. Export to Databases (Optional)

```bash
# Qdrant (semantic search)
python scripts/load_to_databases.py index.scip --qdrant

# Memgraph (graph queries)
python scripts/load_to_databases.py index.scip --memgraph

# Marquez (lineage tracking)
python scripts/load_to_databases.py index.scip --marquez
```

---

## Best Practices

### 1. Automate Indexing

```bash
# Add to CI/CD pipeline
- name: Generate SCIP Index
  run: |
    <language-indexer> index
    curl -X POST $NCODE_SERVER/v1/index/load \
      -d "{\"path\": \"$(pwd)/index.scip\"}"
```

### 2. Version Control

```gitignore
# Add to .gitignore for large projects
*.scip

# Or commit for smaller projects
!index.scip
```

### 3. Incremental Indexing

```bash
# Only re-index changed modules
if git diff --name-only HEAD~1 | grep "src/module"; then
    <indexer> index src/module
fi
```

### 4. Monitor Index Quality

```bash
# Check index stats
curl http://localhost:18003/v1/symbols \
  -d '{"file": "src/main.ext"}' | jq '.symbols | length'
```

---

## Troubleshooting

### Common Issues Across Languages

1. **No symbols found**: Ensure code compiles without errors
2. **Slow indexing**: Use incremental indexing or exclude test files
3. **Memory errors**: Increase heap size for indexer
4. **Path issues**: Use absolute paths when loading to nCode

### Getting Help

- Check [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md)
- Review indexer-specific documentation
- Open issue on [GitHub](https://github.com/sourcegraph/scip)

---

**Last Updated:** 2026-01-17  
**Version:** 1.0  
**Coverage:** 28+ Languages
