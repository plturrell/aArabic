"""
Source Entity Management (Mojo side)
Provides source data structures and business logic
"""

from collections import Dict
from memory import UnsafePointer
from time import now

# ============================================================================
# Source Entity
# ============================================================================

struct SourceType:
    """Source type enumeration"""
    alias URL = 0
    alias PDF = 1
    alias TEXT = 2
    alias FILE = 3
    
    var value: Int32
    
    fn __init__(inout self, value: Int32):
        self.value = value
    
    fn to_string(self) -> String:
        """Convert to string representation"""
        if self.value == Self.URL:
            return "URL"
        elif self.value == Self.PDF:
            return "PDF"
        elif self.value == Self.TEXT:
            return "Text"
        elif self.value == Self.FILE:
            return "File"
        else:
            return "Unknown"
    
    @staticmethod
    fn from_string(s: String) raises -> SourceType:
        """Parse from string"""
        if s == "URL":
            return SourceType(Self.URL)
        elif s == "PDF":
            return SourceType(Self.PDF)
        elif s == "Text" or s == "TEXT":
            return SourceType(Self.TEXT)
        elif s == "File" or s == "FILE":
            return SourceType(Self.FILE)
        else:
            raise Error("Invalid source type: " + s)

struct SourceStatus:
    """Source status enumeration"""
    alias PENDING = 0
    alias PROCESSING = 1
    alias READY = 2
    alias FAILED = 3
    
    var value: Int32
    
    fn __init__(inout self, value: Int32):
        self.value = value
    
    fn to_string(self) -> String:
        """Convert to string representation"""
        if self.value == Self.PENDING:
            return "Pending"
        elif self.value == Self.PROCESSING:
            return "Processing"
        elif self.value == Self.READY:
            return "Ready"
        elif self.value == Self.FAILED:
            return "Failed"
        else:
            return "Unknown"
    
    @staticmethod
    fn from_string(s: String) raises -> SourceStatus:
        """Parse from string"""
        if s == "Pending" or s == "PENDING":
            return SourceStatus(Self.PENDING)
        elif s == "Processing" or s == "PROCESSING":
            return SourceStatus(Self.PROCESSING)
        elif s == "Ready" or s == "READY":
            return SourceStatus(Self.READY)
        elif s == "Failed" or s == "FAILED":
            return SourceStatus(Self.FAILED)
        else:
            raise Error("Invalid status: " + s)

struct Source:
    """Source entity with all fields"""
    var id: String
    var title: String
    var source_type: SourceType
    var url: String
    var content: String
    var status: SourceStatus
    var created_at: String
    var updated_at: String
    
    fn __init__(inout self, 
                id: String,
                title: String, 
                source_type: SourceType,
                url: String,
                content: String,
                status: SourceStatus,
                created_at: String,
                updated_at: String):
        """Initialize a source"""
        self.id = id
        self.title = title
        self.source_type = source_type
        self.url = url
        self.content = content
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
    
    fn validate(self) raises -> Bool:
        """Validate source fields"""
        if len(self.title) == 0:
            raise Error("Title cannot be empty")
        if len(self.url) == 0:
            raise Error("URL cannot be empty")
        if len(self.id) == 0:
            raise Error("ID cannot be empty")
        return True
    
    fn set_status(inout self, status: SourceStatus):
        """Update source status"""
        self.status = status
        self.updated_at = get_iso_timestamp()
    
    fn update_content(inout self, content: String):
        """Update source content"""
        self.content = content
        self.updated_at = get_iso_timestamp()
        self.status = SourceStatus(SourceStatus.READY)

# ============================================================================
# Source Storage (Mojo side)
# ============================================================================

struct SourceStorage:
    """In-memory storage for sources"""
    var sources: Dict[String, Source]
    
    fn __init__(inout self):
        """Initialize empty storage"""
        self.sources = Dict[String, Source]()
    
    fn put(inout self, id: String, source: Source) raises:
        """Store a source"""
        _ = source.validate()  # Validate before storing
        self.sources[id] = source
    
    fn get(self, id: String) raises -> Source:
        """Get source by ID"""
        if id in self.sources:
            return self.sources[id]
        else:
            raise Error("Source not found: " + id)
    
    fn exists(self, id: String) -> Bool:
        """Check if source exists"""
        return id in self.sources
    
    fn delete(inout self, id: String) raises:
        """Delete source by ID"""
        if id in self.sources:
            _ = self.sources.pop(id)
        else:
            raise Error("Source not found: " + id)
    
    fn count(self) -> Int:
        """Get count of sources"""
        return len(self.sources)
    
    fn get_all(self) -> List[Source]:
        """Get all sources as list"""
        var result = List[Source]()
        for item in self.sources.items():
            result.append(item[].value)
        return result

# ============================================================================
# Utility Functions
# ============================================================================

fn generate_source_id() -> String:
    """Generate unique source ID"""
    var timestamp = now()
    var random_part = int(timestamp * 1000) % 1000000
    return "source_" + str(int(timestamp)) + "_" + str(random_part)

fn get_iso_timestamp() -> String:
    """Get current ISO 8601 timestamp"""
    # Simplified for MVP - would use proper datetime formatting
    var timestamp = now()
    return "2026-01-16T13:45:00Z"  # Placeholder

# ============================================================================
# Source Manager
# ============================================================================

struct SourceManager:
    """High-level source management"""
    var storage: SourceStorage
    
    fn __init__(inout self):
        """Initialize source manager"""
        self.storage = SourceStorage()
    
    fn create_source(inout self,
                     title: String,
                     source_type: SourceType,
                     url: String,
                     content: String) raises -> String:
        """Create a new source and return its ID"""
        var id = generate_source_id()
        var timestamp = get_iso_timestamp()
        
        var source = Source(
            id,
            title,
            source_type,
            url,
            content,
            SourceStatus(SourceStatus.READY),
            timestamp,
            timestamp
        )
        
        self.storage.put(id, source)
        return id
    
    fn get_source(self, id: String) raises -> Source:
        """Get source by ID"""
        return self.storage.get(id)
    
    fn delete_source(inout self, id: String) raises:
        """Delete source by ID"""
        self.storage.delete(id)
    
    fn list_sources(self) -> List[Source]:
        """List all sources"""
        return self.storage.get_all()
    
    fn update_source_status(inout self, id: String, status: SourceStatus) raises:
        """Update source status"""
        var source = self.storage.get(id)
        source.set_status(status)
        self.storage.put(id, source)
    
    fn update_source_content(inout self, id: String, content: String) raises:
        """Update source content"""
        var source = self.storage.get(id)
        source.update_content(content)
        self.storage.put(id, source)
