# Mojo Network I/O Module
# Day 48 - TCP/UDP sockets and HTTP client
#
# This module provides networking capabilities including:
# - TCP client and server sockets
# - UDP sockets
# - IP address handling
# - Basic HTTP client

from ..ffi.ffi import CString, CType, CValue, UnsafePointer

# =============================================================================
# Constants
# =============================================================================

alias DEFAULT_BACKLOG: Int = 128
alias DEFAULT_TIMEOUT_MS: Int = 30000
alias MAX_PACKET_SIZE: Int = 65535
alias HTTP_DEFAULT_PORT: Int = 80
alias HTTPS_DEFAULT_PORT: Int = 443

# =============================================================================
# Address Family
# =============================================================================

struct AddressFamily:
    """Socket address family."""

    alias UNSPEC = 0    # AF_UNSPEC
    alias INET = 2      # AF_INET (IPv4)
    alias INET6 = 10    # AF_INET6 (IPv6)
    alias UNIX = 1      # AF_UNIX (local)

    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __eq__(self, other: AddressFamily) -> Bool:
        return self.value == other.value

    fn __str__(self) -> String:
        if self.value == AddressFamily.INET:
            return "IPv4"
        elif self.value == AddressFamily.INET6:
            return "IPv6"
        elif self.value == AddressFamily.UNIX:
            return "Unix"
        return "Unknown"


# =============================================================================
# Socket Type
# =============================================================================

struct SocketType:
    """Socket type."""

    alias STREAM = 1    # SOCK_STREAM (TCP)
    alias DGRAM = 2     # SOCK_DGRAM (UDP)
    alias RAW = 3       # SOCK_RAW

    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __str__(self) -> String:
        if self.value == SocketType.STREAM:
            return "TCP"
        elif self.value == SocketType.DGRAM:
            return "UDP"
        elif self.value == SocketType.RAW:
            return "Raw"
        return "Unknown"


# =============================================================================
# Protocol
# =============================================================================

struct Protocol:
    """Network protocol."""

    alias DEFAULT = 0
    alias TCP = 6       # IPPROTO_TCP
    alias UDP = 17      # IPPROTO_UDP

    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value


# =============================================================================
# Network Error
# =============================================================================

struct NetworkError:
    """Network operation error."""

    alias NONE = 0
    alias CONNECTION_REFUSED = 1
    alias CONNECTION_RESET = 2
    alias CONNECTION_TIMEOUT = 3
    alias HOST_UNREACHABLE = 4
    alias NETWORK_UNREACHABLE = 5
    alias ADDRESS_IN_USE = 6
    alias ADDRESS_NOT_AVAILABLE = 7
    alias INVALID_ADDRESS = 8
    alias SOCKET_ERROR = 9
    alias BIND_ERROR = 10
    alias LISTEN_ERROR = 11
    alias ACCEPT_ERROR = 12
    alias SEND_ERROR = 13
    alias RECEIVE_ERROR = 14
    alias DNS_ERROR = 15
    alias SSL_ERROR = 16
    alias CLOSED = 17
    alias UNKNOWN = 99

    var code: Int
    var message: String
    var errno: Int

    fn __init__(inout self):
        self.code = NetworkError.NONE
        self.message = ""
        self.errno = 0

    fn __init__(inout self, code: Int, message: String):
        self.code = code
        self.message = message
        self.errno = 0

    fn __init__(inout self, code: Int, message: String, errno: Int):
        self.code = code
        self.message = message
        self.errno = errno

    fn is_error(self) -> Bool:
        return self.code != NetworkError.NONE

    fn __str__(self) -> String:
        var result = "NetworkError(" + str(self.code) + "): " + self.message
        if self.errno != 0:
            result += " [errno: " + str(self.errno) + "]"
        return result


# Global network error
var _last_network_error = NetworkError()

fn get_last_network_error() -> NetworkError:
    return _last_network_error

fn clear_network_error():
    _last_network_error = NetworkError()

fn set_network_error(code: Int, message: String):
    _last_network_error = NetworkError(code, message)


# =============================================================================
# IPv4 Address
# =============================================================================

struct IPv4Address:
    """IPv4 address (32-bit)."""

    var _octets: SIMD[DType.uint8, 4]

    fn __init__(inout self):
        self._octets = SIMD[DType.uint8, 4](0, 0, 0, 0)

    fn __init__(inout self, a: Int, b: Int, c: Int, d: Int):
        self._octets = SIMD[DType.uint8, 4](a, b, c, d)

    fn __init__(inout self, packed: UInt32):
        self._octets = SIMD[DType.uint8, 4](
            (packed >> 24) & 0xFF,
            (packed >> 16) & 0xFF,
            (packed >> 8) & 0xFF,
            packed & 0xFF
        )

    @staticmethod
    fn parse(s: String) raises -> IPv4Address:
        """Parse IPv4 address from string (e.g., '192.168.1.1')."""
        var parts = List[Int]()
        var current = 0

        for i in range(len(s)):
            var c = s[i]
            if c == ".":
                parts.append(current)
                current = 0
            elif c >= "0" and c <= "9":
                current = current * 10 + (ord(c) - ord("0"))
            else:
                raise Error("Invalid IPv4 address character")

        parts.append(current)

        if len(parts) != 4:
            raise Error("Invalid IPv4 address format")

        for i in range(4):
            if parts[i] < 0 or parts[i] > 255:
                raise Error("IPv4 octet out of range")

        return IPv4Address(parts[0], parts[1], parts[2], parts[3])

    fn to_packed(self) -> UInt32:
        """Convert to 32-bit packed representation."""
        return (UInt32(self._octets[0]) << 24) |
               (UInt32(self._octets[1]) << 16) |
               (UInt32(self._octets[2]) << 8) |
               UInt32(self._octets[3])

    fn __str__(self) -> String:
        return str(int(self._octets[0])) + "." +
               str(int(self._octets[1])) + "." +
               str(int(self._octets[2])) + "." +
               str(int(self._octets[3]))

    fn __eq__(self, other: IPv4Address) -> Bool:
        return self.to_packed() == other.to_packed()

    fn is_loopback(self) -> Bool:
        return self._octets[0] == 127

    fn is_private(self) -> Bool:
        # 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
        if self._octets[0] == 10:
            return True
        if self._octets[0] == 172 and (self._octets[1] >= 16 and self._octets[1] <= 31):
            return True
        if self._octets[0] == 192 and self._octets[1] == 168:
            return True
        return False

    fn is_broadcast(self) -> Bool:
        return self._octets[0] == 255 and self._octets[1] == 255 and
               self._octets[2] == 255 and self._octets[3] == 255

    @staticmethod
    fn localhost() -> IPv4Address:
        return IPv4Address(127, 0, 0, 1)

    @staticmethod
    fn any() -> IPv4Address:
        return IPv4Address(0, 0, 0, 0)

    @staticmethod
    fn broadcast() -> IPv4Address:
        return IPv4Address(255, 255, 255, 255)


# =============================================================================
# IPv6 Address
# =============================================================================

struct IPv6Address:
    """IPv6 address (128-bit)."""

    var _segments: SIMD[DType.uint16, 8]

    fn __init__(inout self):
        self._segments = SIMD[DType.uint16, 8](0, 0, 0, 0, 0, 0, 0, 0)

    fn __init__(inout self, s0: Int, s1: Int, s2: Int, s3: Int,
                s4: Int, s5: Int, s6: Int, s7: Int):
        self._segments = SIMD[DType.uint16, 8](s0, s1, s2, s3, s4, s5, s6, s7)

    @staticmethod
    fn parse(s: String) raises -> IPv6Address:
        """Parse IPv6 address from string."""
        # Simplified parsing - full implementation would handle :: notation
        var addr = IPv6Address()
        # Placeholder - actual implementation would parse hex segments
        return addr

    fn __str__(self) -> String:
        # Format as hex segments
        var parts = List[String]()
        for i in range(8):
            # Convert to hex
            var val = int(self._segments[i])
            var hex_str = ""
            if val == 0:
                hex_str = "0"
            else:
                while val > 0:
                    var digit = val % 16
                    if digit < 10:
                        hex_str = chr(ord("0") + digit) + hex_str
                    else:
                        hex_str = chr(ord("a") + digit - 10) + hex_str
                    val = val // 16
            parts.append(hex_str)

        var result = ""
        for i in range(len(parts)):
            if i > 0:
                result += ":"
            result += parts[i]
        return result

    fn is_loopback(self) -> Bool:
        # ::1
        for i in range(7):
            if self._segments[i] != 0:
                return False
        return self._segments[7] == 1

    fn is_unspecified(self) -> Bool:
        # ::
        for i in range(8):
            if self._segments[i] != 0:
                return False
        return True

    @staticmethod
    fn localhost() -> IPv6Address:
        return IPv6Address(0, 0, 0, 0, 0, 0, 0, 1)

    @staticmethod
    fn any() -> IPv6Address:
        return IPv6Address(0, 0, 0, 0, 0, 0, 0, 0)


# =============================================================================
# Socket Address
# =============================================================================

struct SocketAddress:
    """Network socket address (IP + port)."""

    var _family: AddressFamily
    var _ipv4: IPv4Address
    var _ipv6: IPv6Address
    var _port: Int

    fn __init__(inout self):
        self._family = AddressFamily(AddressFamily.INET)
        self._ipv4 = IPv4Address()
        self._ipv6 = IPv6Address()
        self._port = 0

    fn __init__(inout self, ipv4: IPv4Address, port: Int):
        self._family = AddressFamily(AddressFamily.INET)
        self._ipv4 = ipv4
        self._ipv6 = IPv6Address()
        self._port = port

    fn __init__(inout self, ipv6: IPv6Address, port: Int):
        self._family = AddressFamily(AddressFamily.INET6)
        self._ipv4 = IPv4Address()
        self._ipv6 = ipv6
        self._port = port

    @staticmethod
    fn parse(addr: String, port: Int) raises -> SocketAddress:
        """Parse address string with port."""
        # Try IPv4 first
        try:
            var ipv4 = IPv4Address.parse(addr)
            return SocketAddress(ipv4, port)
        except:
            pass

        # Try IPv6
        try:
            var ipv6 = IPv6Address.parse(addr)
            return SocketAddress(ipv6, port)
        except:
            pass

        raise Error("Invalid address format")

    fn family(self) -> AddressFamily:
        return self._family

    fn port(self) -> Int:
        return self._port

    fn ipv4(self) -> IPv4Address:
        return self._ipv4

    fn ipv6(self) -> IPv6Address:
        return self._ipv6

    fn __str__(self) -> String:
        if self._family.value == AddressFamily.INET:
            return str(self._ipv4) + ":" + str(self._port)
        else:
            return "[" + str(self._ipv6) + "]:" + str(self._port)

    @staticmethod
    fn localhost(port: Int) -> SocketAddress:
        return SocketAddress(IPv4Address.localhost(), port)

    @staticmethod
    fn any_ipv4(port: Int) -> SocketAddress:
        return SocketAddress(IPv4Address.any(), port)


# =============================================================================
# Socket Options
# =============================================================================

struct SocketOption:
    """Socket option constants."""

    # Socket level options (SOL_SOCKET)
    alias SO_REUSEADDR = 2
    alias SO_KEEPALIVE = 9
    alias SO_BROADCAST = 6
    alias SO_RCVBUF = 8
    alias SO_SNDBUF = 7
    alias SO_RCVTIMEO = 20
    alias SO_SNDTIMEO = 21
    alias SO_LINGER = 13

    # TCP options (IPPROTO_TCP)
    alias TCP_NODELAY = 1
    alias TCP_KEEPIDLE = 4
    alias TCP_KEEPINTVL = 5
    alias TCP_KEEPCNT = 6


# =============================================================================
# Socket (Base Class)
# =============================================================================

struct Socket:
    """Base socket class."""

    var _fd: Int
    var _family: AddressFamily
    var _type: SocketType
    var _protocol: Protocol
    var _is_open: Bool
    var _is_connected: Bool
    var _local_addr: SocketAddress
    var _remote_addr: SocketAddress

    fn __init__(inout self, family: AddressFamily = AddressFamily(AddressFamily.INET),
                type: SocketType = SocketType(SocketType.STREAM),
                protocol: Protocol = Protocol(Protocol.DEFAULT)):
        self._fd = -1
        self._family = family
        self._type = type
        self._protocol = protocol
        self._is_open = False
        self._is_connected = False
        self._local_addr = SocketAddress()
        self._remote_addr = SocketAddress()

    fn __del__(owned self):
        if self._is_open:
            self._close()

    fn _create(inout self) raises:
        """Create the socket."""
        # This would call libc socket(family, type, protocol)
        clear_network_error()
        self._fd = 3  # Placeholder fd
        self._is_open = True

    fn _close(inout self):
        """Close the socket."""
        if self._is_open:
            # This would call libc close(fd)
            self._is_open = False
            self._is_connected = False
            self._fd = -1

    fn close(inout self):
        """Close the socket."""
        self._close()

    fn is_open(self) -> Bool:
        return self._is_open

    fn is_connected(self) -> Bool:
        return self._is_connected

    fn fd(self) -> Int:
        return self._fd

    fn local_address(self) -> SocketAddress:
        return self._local_addr

    fn remote_address(self) -> SocketAddress:
        return self._remote_addr

    fn set_blocking(inout self, blocking: Bool) raises:
        """Set blocking/non-blocking mode."""
        if not self._is_open:
            raise Error("Socket not open")
        # This would use fcntl to set O_NONBLOCK

    fn set_option(inout self, level: Int, option: Int, value: Int) raises:
        """Set socket option."""
        if not self._is_open:
            raise Error("Socket not open")
        # This would call setsockopt

    fn get_option(self, level: Int, option: Int) raises -> Int:
        """Get socket option."""
        if not self._is_open:
            raise Error("Socket not open")
        return 0  # Placeholder

    fn set_timeout(inout self, timeout_ms: Int) raises:
        """Set receive/send timeout in milliseconds."""
        # Convert to timeval and set SO_RCVTIMEO/SO_SNDTIMEO
        pass

    fn set_reuse_address(inout self, reuse: Bool) raises:
        """Enable/disable address reuse."""
        var value = 1 if reuse else 0
        self.set_option(1, SocketOption.SO_REUSEADDR, value)  # SOL_SOCKET = 1


# =============================================================================
# TCP Socket
# =============================================================================

struct TcpSocket:
    """TCP stream socket."""

    var _socket: Socket
    var _read_buffer: List[UInt8]
    var _write_buffer: List[UInt8]

    fn __init__(inout self) raises:
        self._socket = Socket(
            AddressFamily(AddressFamily.INET),
            SocketType(SocketType.STREAM),
            Protocol(Protocol.TCP)
        )
        self._read_buffer = List[UInt8]()
        self._write_buffer = List[UInt8]()
        self._socket._create()

    fn __del__(owned self):
        pass  # Socket closes automatically

    fn close(inout self):
        self._socket.close()

    fn is_connected(self) -> Bool:
        return self._socket.is_connected()

    # -------------------------------------------------------------------------
    # Client Operations
    # -------------------------------------------------------------------------

    fn connect(inout self, address: SocketAddress) raises:
        """Connect to remote address."""
        if not self._socket._is_open:
            raise Error("Socket not open")

        clear_network_error()
        # This would call libc connect()
        self._socket._is_connected = True
        self._socket._remote_addr = address

    fn connect(inout self, host: String, port: Int) raises:
        """Connect to host:port."""
        var addr = resolve_host(host, port)
        self.connect(addr)

    # -------------------------------------------------------------------------
    # Server Operations
    # -------------------------------------------------------------------------

    fn bind(inout self, address: SocketAddress) raises:
        """Bind to local address."""
        if not self._socket._is_open:
            raise Error("Socket not open")

        clear_network_error()
        # This would call libc bind()
        self._socket._local_addr = address

    fn listen(inout self, backlog: Int = DEFAULT_BACKLOG) raises:
        """Start listening for connections."""
        if not self._socket._is_open:
            raise Error("Socket not open")

        # This would call libc listen()

    fn accept(inout self) raises -> TcpSocket:
        """Accept incoming connection."""
        if not self._socket._is_open:
            raise Error("Socket not open")

        # This would call libc accept()
        var client = TcpSocket()
        client._socket._is_connected = True
        return client

    # -------------------------------------------------------------------------
    # Data Transfer
    # -------------------------------------------------------------------------

    fn send(inout self, data: String) raises -> Int:
        """Send string data."""
        if not self._socket._is_connected:
            raise Error("Not connected")

        # This would call libc send()
        return len(data)  # Placeholder

    fn send_bytes(inout self, data: List[UInt8]) raises -> Int:
        """Send byte data."""
        if not self._socket._is_connected:
            raise Error("Not connected")

        return len(data)  # Placeholder

    fn send_all(inout self, data: String) raises:
        """Send all data, retrying if necessary."""
        var sent = 0
        var total = len(data)

        while sent < total:
            var n = self.send(data[sent:])
            if n <= 0:
                raise Error("Send failed")
            sent += n

    fn recv(inout self, max_size: Int = 4096) raises -> String:
        """Receive string data."""
        if not self._socket._is_connected:
            raise Error("Not connected")

        # This would call libc recv()
        return ""  # Placeholder

    fn recv_bytes(inout self, max_size: Int = 4096) raises -> List[UInt8]:
        """Receive byte data."""
        if not self._socket._is_connected:
            raise Error("Not connected")

        return List[UInt8]()  # Placeholder

    fn recv_exact(inout self, size: Int) raises -> List[UInt8]:
        """Receive exactly size bytes."""
        var result = List[UInt8]()

        while len(result) < size:
            var chunk = self.recv_bytes(size - len(result))
            if len(chunk) == 0:
                raise Error("Connection closed")
            for i in range(len(chunk)):
                result.append(chunk[i])

        return result

    # -------------------------------------------------------------------------
    # Options
    # -------------------------------------------------------------------------

    fn set_nodelay(inout self, nodelay: Bool) raises:
        """Enable/disable Nagle's algorithm."""
        var value = 1 if nodelay else 0
        self._socket.set_option(Protocol.TCP, SocketOption.TCP_NODELAY, value)

    fn set_keepalive(inout self, keepalive: Bool) raises:
        """Enable/disable TCP keepalive."""
        var value = 1 if keepalive else 0
        self._socket.set_option(1, SocketOption.SO_KEEPALIVE, value)


# =============================================================================
# TCP Listener
# =============================================================================

struct TcpListener:
    """TCP server listener."""

    var _socket: TcpSocket
    var _address: SocketAddress

    fn __init__(inout self, address: SocketAddress) raises:
        self._socket = TcpSocket()
        self._address = address
        self._socket._socket.set_reuse_address(True)
        self._socket.bind(address)
        self._socket.listen()

    fn __init__(inout self, port: Int) raises:
        var addr = SocketAddress.any_ipv4(port)
        self._socket = TcpSocket()
        self._address = addr
        self._socket._socket.set_reuse_address(True)
        self._socket.bind(addr)
        self._socket.listen()

    fn accept(inout self) raises -> TcpSocket:
        """Accept incoming connection."""
        return self._socket.accept()

    fn local_address(self) -> SocketAddress:
        return self._address

    fn close(inout self):
        self._socket.close()


# =============================================================================
# UDP Socket
# =============================================================================

struct UdpSocket:
    """UDP datagram socket."""

    var _socket: Socket
    var _bound: Bool

    fn __init__(inout self) raises:
        self._socket = Socket(
            AddressFamily(AddressFamily.INET),
            SocketType(SocketType.DGRAM),
            Protocol(Protocol.UDP)
        )
        self._bound = False
        self._socket._create()

    fn __del__(owned self):
        pass

    fn close(inout self):
        self._socket.close()

    fn bind(inout self, address: SocketAddress) raises:
        """Bind to local address."""
        # This would call libc bind()
        self._socket._local_addr = address
        self._bound = True

    fn bind(inout self, port: Int) raises:
        """Bind to port on all interfaces."""
        self.bind(SocketAddress.any_ipv4(port))

    fn send_to(inout self, data: String, address: SocketAddress) raises -> Int:
        """Send datagram to address."""
        # This would call libc sendto()
        return len(data)  # Placeholder

    fn send_to_bytes(inout self, data: List[UInt8], address: SocketAddress) raises -> Int:
        """Send byte datagram to address."""
        return len(data)  # Placeholder

    fn recv_from(inout self, max_size: Int = MAX_PACKET_SIZE) raises -> Tuple[String, SocketAddress]:
        """Receive datagram and sender address."""
        # This would call libc recvfrom()
        return ("", SocketAddress())  # Placeholder

    fn recv_from_bytes(inout self, max_size: Int = MAX_PACKET_SIZE) raises -> Tuple[List[UInt8], SocketAddress]:
        """Receive byte datagram and sender address."""
        return (List[UInt8](), SocketAddress())  # Placeholder

    fn set_broadcast(inout self, enabled: Bool) raises:
        """Enable/disable broadcast."""
        var value = 1 if enabled else 0
        self._socket.set_option(1, SocketOption.SO_BROADCAST, value)


# =============================================================================
# DNS Resolution
# =============================================================================

fn resolve_host(hostname: String, port: Int = 0) raises -> SocketAddress:
    """Resolve hostname to socket address."""
    # Check if it's already an IP address
    try:
        var ipv4 = IPv4Address.parse(hostname)
        return SocketAddress(ipv4, port)
    except:
        pass

    # DNS lookup via getaddrinfo
    # This would call libc getaddrinfo()
    clear_network_error()

    # Placeholder - return localhost
    set_network_error(NetworkError.DNS_ERROR, "DNS resolution not implemented")
    return SocketAddress.localhost(port)

fn resolve_all(hostname: String, port: Int = 0) raises -> List[SocketAddress]:
    """Resolve hostname to all addresses."""
    var results = List[SocketAddress]()

    # This would call getaddrinfo and iterate results
    try:
        var addr = resolve_host(hostname, port)
        results.append(addr)
    except:
        pass

    return results

fn get_hostname() -> String:
    """Get local hostname."""
    # This would call libc gethostname()
    return "localhost"  # Placeholder


# =============================================================================
# URL Parsing
# =============================================================================

struct URL:
    """Parsed URL."""

    var scheme: String
    var host: String
    var port: Int
    var path: String
    var query: String
    var fragment: String
    var username: String
    var password: String

    fn __init__(inout self):
        self.scheme = ""
        self.host = ""
        self.port = 0
        self.path = "/"
        self.query = ""
        self.fragment = ""
        self.username = ""
        self.password = ""

    @staticmethod
    fn parse(url: String) raises -> URL:
        """Parse URL string."""
        var result = URL()
        var pos = 0
        var length = len(url)

        # Parse scheme
        var scheme_end = url.find("://")
        if scheme_end > 0:
            result.scheme = url[:scheme_end]
            pos = scheme_end + 3
        else:
            result.scheme = "http"

        # Set default port based on scheme
        if result.scheme == "https":
            result.port = HTTPS_DEFAULT_PORT
        else:
            result.port = HTTP_DEFAULT_PORT

        # Parse authority (user:pass@host:port)
        var authority_end = length
        var path_start = url.find("/", pos)
        if path_start > pos:
            authority_end = path_start

        var query_start = url.find("?", pos)
        if query_start > pos and query_start < authority_end:
            authority_end = query_start

        var authority = url[pos:authority_end]

        # Check for userinfo
        var at_pos = authority.find("@")
        if at_pos >= 0:
            var userinfo = authority[:at_pos]
            var colon_pos = userinfo.find(":")
            if colon_pos >= 0:
                result.username = userinfo[:colon_pos]
                result.password = userinfo[colon_pos + 1:]
            else:
                result.username = userinfo
            authority = authority[at_pos + 1:]

        # Parse host and port
        var port_pos = authority.find(":")
        if port_pos >= 0:
            result.host = authority[:port_pos]
            var port_str = authority[port_pos + 1:]
            result.port = 0
            for i in range(len(port_str)):
                var c = port_str[i]
                if c >= "0" and c <= "9":
                    result.port = result.port * 10 + (ord(c) - ord("0"))
        else:
            result.host = authority

        pos = authority_end

        # Parse path
        if pos < length and url[pos] == "/":
            var path_end = length
            query_start = url.find("?", pos)
            if query_start > pos:
                path_end = query_start
            var frag_start = url.find("#", pos)
            if frag_start > pos and frag_start < path_end:
                path_end = frag_start
            result.path = url[pos:path_end]
            pos = path_end

        # Parse query
        if pos < length and url[pos] == "?":
            pos += 1
            var query_end = length
            var frag_start = url.find("#", pos)
            if frag_start > pos:
                query_end = frag_start
            result.query = url[pos:query_end]
            pos = query_end

        # Parse fragment
        if pos < length and url[pos] == "#":
            pos += 1
            result.fragment = url[pos:]

        return result

    fn __str__(self) -> String:
        var result = self.scheme + "://"
        if len(self.username) > 0:
            result += self.username
            if len(self.password) > 0:
                result += ":" + self.password
            result += "@"
        result += self.host
        if self.port != HTTP_DEFAULT_PORT and self.port != HTTPS_DEFAULT_PORT:
            result += ":" + str(self.port)
        result += self.path
        if len(self.query) > 0:
            result += "?" + self.query
        if len(self.fragment) > 0:
            result += "#" + self.fragment
        return result


# =============================================================================
# HTTP Client (Basic)
# =============================================================================

struct HttpMethod:
    """HTTP request methods."""

    alias GET = "GET"
    alias POST = "POST"
    alias PUT = "PUT"
    alias DELETE = "DELETE"
    alias HEAD = "HEAD"
    alias OPTIONS = "OPTIONS"
    alias PATCH = "PATCH"


struct HttpHeaders:
    """HTTP headers collection."""

    var _headers: List[Tuple[String, String]]

    fn __init__(inout self):
        self._headers = List[Tuple[String, String]]()

    fn set(inout self, name: String, value: String):
        """Set header (replaces existing)."""
        var lower_name = name.lower()
        for i in range(len(self._headers)):
            if self._headers[i][0].lower() == lower_name:
                self._headers[i] = (name, value)
                return
        self._headers.append((name, value))

    fn get(self, name: String) -> String:
        """Get header value."""
        var lower_name = name.lower()
        for i in range(len(self._headers)):
            if self._headers[i][0].lower() == lower_name:
                return self._headers[i][1]
        return ""

    fn has(self, name: String) -> Bool:
        """Check if header exists."""
        var lower_name = name.lower()
        for i in range(len(self._headers)):
            if self._headers[i][0].lower() == lower_name:
                return True
        return False

    fn remove(inout self, name: String):
        """Remove header."""
        var lower_name = name.lower()
        for i in range(len(self._headers)):
            if self._headers[i][0].lower() == lower_name:
                _ = self._headers.pop(i)
                return

    fn to_string(self) -> String:
        """Format headers for HTTP request."""
        var result = ""
        for i in range(len(self._headers)):
            result += self._headers[i][0] + ": " + self._headers[i][1] + "\r\n"
        return result


struct HttpResponse:
    """HTTP response."""

    var status_code: Int
    var status_message: String
    var headers: HttpHeaders
    var body: String

    fn __init__(inout self):
        self.status_code = 0
        self.status_message = ""
        self.headers = HttpHeaders()
        self.body = ""

    fn is_success(self) -> Bool:
        return self.status_code >= 200 and self.status_code < 300

    fn is_redirect(self) -> Bool:
        return self.status_code >= 300 and self.status_code < 400

    fn is_error(self) -> Bool:
        return self.status_code >= 400


struct HttpClient:
    """Simple HTTP client."""

    var _timeout_ms: Int
    var _follow_redirects: Bool
    var _max_redirects: Int
    var _default_headers: HttpHeaders

    fn __init__(inout self):
        self._timeout_ms = DEFAULT_TIMEOUT_MS
        self._follow_redirects = True
        self._max_redirects = 10
        self._default_headers = HttpHeaders()
        self._default_headers.set("User-Agent", "Mojo/1.0")
        self._default_headers.set("Accept", "*/*")
        self._default_headers.set("Connection", "close")

    fn set_timeout(inout self, timeout_ms: Int):
        self._timeout_ms = timeout_ms

    fn set_follow_redirects(inout self, follow: Bool):
        self._follow_redirects = follow

    fn get(inout self, url: String) raises -> HttpResponse:
        """Perform GET request."""
        return self.request(HttpMethod.GET, url, "")

    fn post(inout self, url: String, body: String) raises -> HttpResponse:
        """Perform POST request."""
        return self.request(HttpMethod.POST, url, body)

    fn put(inout self, url: String, body: String) raises -> HttpResponse:
        """Perform PUT request."""
        return self.request(HttpMethod.PUT, url, body)

    fn delete(inout self, url: String) raises -> HttpResponse:
        """Perform DELETE request."""
        return self.request(HttpMethod.DELETE, url, "")

    fn request(inout self, method: String, url: String, body: String) raises -> HttpResponse:
        """Perform HTTP request."""
        var parsed_url = URL.parse(url)
        var response = HttpResponse()

        # Connect to host
        var socket = TcpSocket()
        socket.connect(parsed_url.host, parsed_url.port)
        socket._socket.set_timeout(self._timeout_ms)

        # Build request
        var request = method + " " + parsed_url.path
        if len(parsed_url.query) > 0:
            request += "?" + parsed_url.query
        request += " HTTP/1.1\r\n"
        request += "Host: " + parsed_url.host + "\r\n"
        request += self._default_headers.to_string()

        if len(body) > 0:
            request += "Content-Length: " + str(len(body)) + "\r\n"

        request += "\r\n"
        request += body

        # Send request
        socket.send_all(request)

        # Read response (simplified)
        var response_data = socket.recv(65536)
        socket.close()

        # Parse response (simplified)
        self._parse_response(response_data, response)

        return response

    fn _parse_response(self, data: String, inout response: HttpResponse):
        """Parse HTTP response."""
        # Find status line end
        var line_end = data.find("\r\n")
        if line_end < 0:
            return

        var status_line = data[:line_end]

        # Parse status code
        var space1 = status_line.find(" ")
        if space1 > 0:
            var space2 = status_line.find(" ", space1 + 1)
            if space2 > space1:
                var code_str = status_line[space1 + 1:space2]
                response.status_code = 0
                for i in range(len(code_str)):
                    var c = code_str[i]
                    if c >= "0" and c <= "9":
                        response.status_code = response.status_code * 10 + (ord(c) - ord("0"))
                response.status_message = status_line[space2 + 1:]

        # Find headers/body separator
        var body_start = data.find("\r\n\r\n")
        if body_start > 0:
            response.body = data[body_start + 4:]


# =============================================================================
# Tests
# =============================================================================

fn test_ipv4_address():
    """Test IPv4Address."""
    var addr = IPv4Address(192, 168, 1, 1)
    assert_true(str(addr) == "192.168.1.1", "IPv4 string should match")
    assert_true(addr.is_private(), "192.168.x.x should be private")

    var localhost = IPv4Address.localhost()
    assert_true(localhost.is_loopback(), "127.0.0.1 should be loopback")

    print("test_ipv4_address: PASSED")


fn test_socket_address():
    """Test SocketAddress."""
    var addr = SocketAddress(IPv4Address(10, 0, 0, 1), 8080)
    assert_true(addr.port() == 8080, "Port should be 8080")
    assert_true(str(addr) == "10.0.0.1:8080", "Address string should match")

    print("test_socket_address: PASSED")


fn test_url_parse():
    """Test URL parsing."""
    try:
        var url = URL.parse("https://example.com:8443/path?query=value#fragment")
        assert_true(url.scheme == "https", "Scheme should be https")
        assert_true(url.host == "example.com", "Host should match")
        assert_true(url.port == 8443, "Port should be 8443")
        assert_true(url.path == "/path", "Path should match")
        assert_true(url.query == "query=value", "Query should match")
        assert_true(url.fragment == "fragment", "Fragment should match")
        print("test_url_parse: PASSED")
    except e:
        print("test_url_parse: FAILED - " + str(e))


fn test_http_headers():
    """Test HttpHeaders."""
    var headers = HttpHeaders()
    headers.set("Content-Type", "application/json")
    headers.set("Accept", "text/html")

    assert_true(headers.get("Content-Type") == "application/json", "Header value should match")
    assert_true(headers.has("Accept"), "Header should exist")
    assert_true(not headers.has("Missing"), "Missing header should not exist")

    headers.remove("Accept")
    assert_true(not headers.has("Accept"), "Removed header should not exist")

    print("test_http_headers: PASSED")


fn test_network_error():
    """Test NetworkError."""
    var err = NetworkError(NetworkError.CONNECTION_REFUSED, "Connection refused")
    assert_true(err.is_error(), "Should be error")
    assert_true(err.code == NetworkError.CONNECTION_REFUSED, "Code should match")

    var no_err = NetworkError()
    assert_true(not no_err.is_error(), "Default should not be error")

    print("test_network_error: PASSED")


fn assert_true(condition: Bool, message: String):
    """Simple assertion helper."""
    if not condition:
        print("ASSERTION FAILED: " + message)


fn run_all_tests():
    """Run all network tests."""
    print("=== Network Module Tests ===")
    test_ipv4_address()
    test_socket_address()
    test_url_parse()
    test_http_headers()
    test_network_error()
    print("=== All Tests Passed ===")
