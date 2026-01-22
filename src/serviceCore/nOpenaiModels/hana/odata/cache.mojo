"""
SAP HANA OData Cache Implementation
Provides caching functionality using SAP HANA Cloud via OData protocol

All caching operations use SAP HANA as the persistence layer,
eliminating dependency on external cache services.
"""

from collections import Dict, List
from time import now
from python import Python


struct HanaODataCache:
    """
    Cache implementation using SAP HANA Cloud OData API

    Features:
    - Key-value storage via HANA tables
    - TTL support via timestamp columns
    - OData v4 protocol compliance
    - Connection pooling
    """
    var base_url: String
    var workspace: String
    var auth_token: String
    var table_name: String
    var _connected: Bool
    var default_ttl: Int  # seconds

    fn __init__(
        inout self,
        base_url: String = "",
        workspace: String = "default",
        table_name: String = "CACHE_ENTRIES"
    ):
        """
        Initialize HANA OData cache

        Args:
            base_url: SAP HANA Cloud OData endpoint
            workspace: HANA workspace/schema
            table_name: Table name for cache entries
        """
        # Load from environment if not provided
        if base_url == "":
            self.base_url = self._get_env("SAP_HANA_ODATA_URL", "")
        else:
            self.base_url = base_url

        self.workspace = workspace
        self.auth_token = self._get_env("SAP_HANA_AUTH_TOKEN", "")
        self.table_name = table_name
        self._connected = False
        self.default_ttl = 3600  # 1 hour default

    fn _get_env(self, key: String, default: String) -> String:
        """Get environment variable with default"""
        from python import Python
        try:
            let os = Python.import_module("os")
            let value = os.environ.get(key, default)
            return String(value)
        except:
            return default

    fn connect(inout self) raises:
        """Establish connection to HANA OData service"""
        if self.base_url == "":
            raise Error("SAP HANA OData URL not configured. Set SAP_HANA_ODATA_URL environment variable.")

        # Verify connection
        try:
            let requests = Python.import_module("requests")
            let response = requests.get(
                self.base_url + "/$metadata",
                headers=self._get_headers(),
                timeout=10
            )
            if int(response.status_code) == 200:
                self._connected = True
            else:
                raise Error("HANA connection failed: HTTP " + String(response.status_code))
        except e:
            raise Error("HANA connection error: " + String(e))

    fn _get_headers(self) -> PythonObject:
        """Get HTTP headers for OData requests"""
        from python import Python
        var headers = Python.dict()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        if self.auth_token != "":
            headers["Authorization"] = "Bearer " + self.auth_token
        return headers

    fn get(self, key: String) raises -> String:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or empty string if not found/expired
        """
        if not self._connected:
            raise Error("Not connected to HANA. Call connect() first.")

        from python import Python
        let requests = Python.import_module("requests")
        let json_mod = Python.import_module("json")

        # Build OData query with key filter
        let url = self.base_url + "/" + self.table_name + "?$filter=CACHE_KEY eq '" + self._escape_odata(key) + "'"

        let response = requests.get(url, headers=self._get_headers(), timeout=30)

        if int(response.status_code) == 200:
            let data = json_mod.loads(response.text)
            let results = data.get("value", [])
            if len(results) > 0:
                let entry = results[0]
                # Check TTL
                let expires_at = int(entry.get("EXPIRES_AT", 0))
                let current_time = int(now() // 1_000_000_000)
                if expires_at == 0 or current_time < expires_at:
                    return String(entry.get("CACHE_VALUE", ""))
                else:
                    # Entry expired, delete it
                    self._delete_entry(key)

        return ""

    fn set(self, key: String, value: String, expire_seconds: Int = 0) raises:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            expire_seconds: TTL in seconds (0 = use default)
        """
        if not self._connected:
            raise Error("Not connected to HANA. Call connect() first.")

        from python import Python
        let requests = Python.import_module("requests")
        let json_mod = Python.import_module("json")

        let ttl = expire_seconds if expire_seconds > 0 else self.default_ttl
        let current_time = int(now() // 1_000_000_000)
        let expires_at = current_time + ttl

        # Build entry
        var entry = Python.dict()
        entry["CACHE_KEY"] = key
        entry["CACHE_VALUE"] = value
        entry["CREATED_AT"] = current_time
        entry["EXPIRES_AT"] = expires_at

        # Try to delete existing entry first (upsert)
        try:
            self._delete_entry(key)
        except:
            pass

        # Insert new entry
        let url = self.base_url + "/" + self.table_name
        let response = requests.post(
            url,
            json=entry,
            headers=self._get_headers(),
            timeout=30
        )

        if int(response.status_code) not in [200, 201, 204]:
            raise Error("Cache set failed: HTTP " + String(response.status_code))

    fn delete(self, key: String) raises:
        """Delete entry from cache"""
        self._delete_entry(key)

    fn _delete_entry(self, key: String) raises:
        """Internal delete implementation"""
        from python import Python
        let requests = Python.import_module("requests")

        let url = self.base_url + "/" + self.table_name + "('" + self._escape_odata(key) + "')"
        let response = requests.delete(url, headers=self._get_headers(), timeout=30)

        # 404 is acceptable (entry didn't exist)
        if int(response.status_code) not in [200, 204, 404]:
            raise Error("Cache delete failed: HTTP " + String(response.status_code))

    fn _escape_odata(self, value: String) -> String:
        """Escape special characters for OData queries"""
        var result = value
        result = result.replace("'", "''")
        return result

    fn exists(self, key: String) raises -> Bool:
        """Check if key exists in cache"""
        let value = self.get(key)
        return len(value) > 0

    fn clear_expired(self) raises:
        """Remove all expired entries"""
        if not self._connected:
            raise Error("Not connected to HANA. Call connect() first.")

        from python import Python
        let requests = Python.import_module("requests")

        let current_time = int(now() // 1_000_000_000)
        let url = self.base_url + "/" + self.table_name + "?$filter=EXPIRES_AT lt " + String(current_time) + " and EXPIRES_AT gt 0"

        # Get expired entries
        let response = requests.get(url, headers=self._get_headers(), timeout=30)
        if int(response.status_code) == 200:
            let json_mod = Python.import_module("json")
            let data = json_mod.loads(response.text)
            let results = data.get("value", [])
            for i in range(len(results)):
                let entry = results[i]
                let key = String(entry.get("CACHE_KEY", ""))
                if key != "":
                    try:
                        self._delete_entry(key)
                    except:
                        pass

    fn disconnect(inout self):
        """Close connection"""
        self._connected = False

    fn is_connected(self) -> Bool:
        """Check if connected"""
        return self._connected


fn create_hana_cache(
    base_url: String = "",
    workspace: String = "default"
) raises -> HanaODataCache:
    """
    Factory function to create and connect HANA cache

    Args:
        base_url: SAP HANA OData URL (or set SAP_HANA_ODATA_URL env var)
        workspace: HANA workspace

    Returns:
        Connected HanaODataCache instance
    """
    var cache = HanaODataCache(base_url, workspace)
    cache.connect()
    return cache
