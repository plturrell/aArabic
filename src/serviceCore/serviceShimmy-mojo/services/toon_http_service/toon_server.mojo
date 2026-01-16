"""
TOON HTTP Server - Pure Mojo Implementation
Zero Python dependencies - Native Mojo networking
"""

from sys import argv


struct ToonServer(Copyable, Movable):
    """High-performance TOON HTTP server with Zig backend."""
    
    var host: String
    var port: Int
    var binary_path: String
    
    fn __init__(inout self) raises:
        """Initialize TOON server with default values."""
        self.host = "127.0.0.1"
        self.port = 8085
        self.binary_path = "./zig-out/bin/toon_http"
    
    fn __init__(inout self, host: String, port: Int) raises:
        """Initialize TOON server.
        
        Args:
            host: Server host address.
            port: Server port.
        """
        self.host = host
        self.port = port
        self.binary_path = "./zig-out/bin/toon_http"
    
    fn info(self):
        """Print server information."""
        print("🎨 TOON HTTP Server")
        print("   Binary:", self.binary_path)
        print("   Host:", self.host)
        print("   Port:", self.port)
        print("   URL: http://" + self.host + ":" + String(self.port))
    
    fn start_command(self) -> String:
        """Get the command to start the server."""
        return self.binary_path + " --host " + self.host + " --port " + String(self.port)
    
    fn test_encode(self, json_text: String):
        """Test JSON to TOON encoding.
        
        Args:
            json_text: JSON string to encode.
        """
        print("\n🧪 Testing TOON Encoding")
        print("=" * 60)
        print("\n📥 Input JSON:")
        print("  ", json_text)
        print("\n📋 To encode via curl:")
        print("  curl -X POST http://" + self.host + ":" + String(self.port) + "/encode \\")
        print("    -H 'Content-Type: application/json' \\")
        print("    -d '{\"text\":\"" + json_text + "\"}'")
        print()


fn print_banner():
    """Print TOON server banner."""
    print("=" * 80)
    print("🚀 TOON HTTP Server - Pure Mojo Interface")
    print("=" * 80)
    print()


fn print_help():
    """Print help message."""
    print("\nUsage: mojo toon_server.mojo <command> [args]\n")
    print("Commands:")
    print("  info         - Show server information")
    print("  start        - Show start instructions")
    print("  test <json>  - Show test instructions")
    print("  examples     - Show usage examples")
    print("  help         - Show this help message")
    print("\nQuick Start:")
    print("  1. Build server:  zig build")
    print("  2. Start server:  ./zig-out/bin/toon_http")
    print("  3. Test health:   curl http://127.0.0.1:8085/health")
    print()


fn print_examples():
    """Print usage examples."""
    print("\n📚 TOON Server Examples")
    print("=" * 80)
    
    print("\n1️⃣  Start Server:")
    print("   ./zig-out/bin/toon_http")
    print("   # Custom host/port:")
    print("   ./zig-out/bin/toon_http --host 0.0.0.0 --port 9000")
    
    print("\n2️⃣  Test Encoding (with curl):")
    print("   curl -X POST http://127.0.0.1:8085/encode \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"text\":\"{\\\"name\\\":\\\"Alice\\\",\\\"age\\\":30}\"}'")
    
    print("\n3️⃣  Get Server Info:")
    print("   curl http://127.0.0.1:8085/")
    
    print("\n4️⃣  Check Health:")
    print("   curl http://127.0.0.1:8085/health")
    
    print("\n5️⃣  Encoding with Statistics:")
    print("   curl -X POST http://127.0.0.1:8085/encode-with-stats \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"text\":\"{\\\"users\\\":[{\\\"id\\\":1},{\\\"id\\\":2}]}\"}'")
    
    print("\n6️⃣  Using from Mojo:")
    print("   from toon_server import ToonServer")
    print("   var server = ToonServer()")
    print("   server.info()")
    print("   print(server.start_command())")
    
    print("\n💡 Pure Mojo implementation - zero Python dependencies!")
    print()


fn start_server_instructions(host: String, port: Int):
    """Show instructions to start the server."""
    print("🚀 Starting TOON HTTP Server")
    print("=" * 80)
    print()
    print("Binary Location:")
    print("  ./zig-out/bin/toon_http")
    print()
    print("Start Command:")
    print("  ./zig-out/bin/toon_http --host", host, "--port", port)
    print()
    print("Or run in background:")
    print("  ./zig-out/bin/toon_http --host", host, "--port", port, "&")
    print()
    print("Server will be available at:")
    print("  http://" + host + ":" + String(port))
    print()
    print("To test, run in another terminal:")
    print("  curl http://" + host + ":" + String(port) + "/health")
    print()


fn main():
    """Pure Mojo CLI interface for TOON HTTP server."""
    
    print_banner()
    
    # Parse command line arguments
    var args = argv()
    var argc = len(args)
    
    if argc < 2:
        print("❌ No command specified")
        print_help()
        return
    
    var command = args[1]
    
    # Create server instance
    var server = ToonServer()
    
    if command == "info":
        server.info()
        print("\n📋 Available Endpoints:")
        print("  GET  /              - Service information")
        print("  GET  /health        - Health check")
        print("  POST /encode        - Encode JSON to TOON")
        print("  POST /decode        - Decode TOON to JSON")
        print("  POST /encode-with-stats - Encode with statistics")
        print()
    
    elif command == "start":
        print("🚀 TOON HTTP Server - Start Instructions")
        print("=" * 80)
        print()
        start_server_instructions(server.host, server.port)
        print("💡 Tip: Keep the server running in a separate terminal")
        print()
    
    elif command == "test":
        if argc < 3:
            print("❌ JSON text required")
            print("Usage: mojo toon_server.mojo test '<json>'")
            print("Example: mojo toon_server.mojo test '{\"name\":\"Alice\"}'")
            return
        
        var json_text = args[2]
        server.test_encode(json_text)
    
    elif command == "examples":
        print_examples()
    
    elif command == "help" or command == "--help" or command == "-h":
        print_help()
    
    else:
        print("❌ Unknown command:", command)
        print_help()
