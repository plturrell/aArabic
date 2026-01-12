"""
DEPRECATED: This file is kept for backward compatibility only.
Please use: python -m backend.api.server
Or see MIGRATION.md for details.

This file will be removed in a future version.
"""

import warnings
import sys
from pathlib import Path

warnings.warn(
    "model_server.py is deprecated. Use 'python -m backend.api.server' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Add backend to path if needed
backend_path = Path(__file__).parent / "backend"
if str(backend_path.parent) not in sys.path:
    sys.path.insert(0, str(backend_path.parent))

# Re-export for backward compatibility
try:
    from backend.api.server import app
except ImportError as e:
    raise ImportError(
        f"Failed to import from new structure: {e}\n"
        "Please use: python -m backend.api.server"
    ) from e

if __name__ == "__main__":
    import uvicorn
    from backend.config.settings import settings
    print("\n⚠️  DEPRECATED: model_server.py is deprecated.")
    print("👉 Use: python -m backend.api.server")
    print("👉 Or: ./scripts/start_backend.sh")
    print("👉 This file will be removed in a future version.\n")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
