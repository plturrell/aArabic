"""
Time utilities for Shimmy-Mojo services.
Avoids Python and uses libc time() via Mojo FFI.
"""

from sys.ffi import external_call


fn unix_timestamp() -> Int:
    """Return Unix epoch seconds."""
    return Int(external_call["time", Int64]())
