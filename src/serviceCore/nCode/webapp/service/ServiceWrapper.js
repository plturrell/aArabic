sap.ui.define([], function () {
    "use strict";

    // Configuration
    const BASE_URL = "http://localhost:18003";
    const USE_MOCK = true; // Set to false to try connecting to real server

    // Mock Data
    const MOCK_FILES = [
        { id: "1", name: "main.mojo", type: "Mojo Source", size: "2KB", status: "saved", path: "src/main.mojo", content: "fn main():\n    print('Hello from Mojo!')" },
        { id: "2", name: "server.zig", type: "Zig Source", size: "14KB", status: "modified", path: "src/server.zig", content: "const std = @import(\"std\");\n\npub fn main() !void {\n    std.debug.print(\"Hello Zig!\", .{});\n}" },
        { id: "3", name: "README.md", type: "Markdown", size: "1KB", status: "saved", path: "README.md", content: "# nCode\nCode Intelligence Server" },
        { id: "4", name: "package.json", type: "JSON", size: "500B", status: "saved", path: "package.json", content: "{\n  \"name\": \"ncode\",\n  \"version\": \"1.0.0\"\n}" },
        { id: "5", name: "utils.js", type: "JavaScript", size: "4KB", status: "new", path: "src/utils.js", content: "console.log('Utils');" }
    ];

    return {
        /**
         * Fetch health status
         */
        checkHealth: function() {
            if (USE_MOCK) return Promise.resolve(true);
            return fetch(BASE_URL + "/health")
                .then(res => res.ok)
                .catch(() => false);
        },

        /**
         * Get list of files in project
         */
        getFiles: function () {
            if (USE_MOCK) {
                return new Promise(resolve => {
                    setTimeout(() => resolve([...MOCK_FILES]), 200);
                });
            }
            // If server supported file listing:
            // return fetch(BASE_URL + "/v1/files").then(res => res.json());
            return Promise.resolve([...MOCK_FILES]); // Fallback
        },

        /**
         * Get file details and content
         */
        getFile: function (sId) {
            if (USE_MOCK) {
                return new Promise((resolve, reject) => {
                    const oFile = MOCK_FILES.find(f => f.id === sId);
                    if (oFile) {
                        setTimeout(() => resolve({ ...oFile }), 200);
                    } else {
                        reject("File not found");
                    }
                });
            }
            // Real implementation would fetch file content
             return Promise.resolve(MOCK_FILES.find(f => f.id === sId));
        },
        
        /**
         * Get symbols for a file (Live feature)
         */
        getSymbols: function (sFilePath) {
            if (USE_MOCK) return Promise.resolve([]);
            
            return fetch(BASE_URL + "/v1/document-symbols", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ file: sFilePath })
            })
            .then(res => res.json())
            .then(data => data.symbols || [])
            .catch(err => {
                console.error("Symbol fetch failed", err);
                return [];
            });
        }
    };
});
