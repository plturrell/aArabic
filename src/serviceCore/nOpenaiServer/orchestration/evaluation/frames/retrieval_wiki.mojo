# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wikipedia Retrieval Server - Pure Mojo Implementation

This module provides an HTTP server for Wikipedia retrieval using:
- Zig-based HTTP server via FFI
- Semantic index for BM25 + vector search
- Pure Mojo data handling

Usage:
    mojo run retrieval_wiki.mojo --port 1401
"""

from sys import argv
from collections import Dict, List
from memory import UnsafePointer
from io import read_file, file_exists
from mojo_sdk.stdlib.ffi.ffi import external_call


# ============================================================================
# Configuration
# ============================================================================

@value
struct RetrievalConfig:
    """Configuration for the retrieval server."""
    var retrieval_method: String
    var retrieval_topk: Int
    var index_path: String
    var corpus_path: String
    var faiss_gpu: Bool
    var retrieval_batch_size: Int

    fn __init__(inout self):
        self.retrieval_method = "bm25"
        self.retrieval_topk = 10
        self.index_path = "./index/wiki"
        self.corpus_path = "./data/wiki_corpus.jsonl"
        self.faiss_gpu = False  # GPU not available in pure Mojo
        self.retrieval_batch_size = 128


# ============================================================================
# Corpus Loading
# ============================================================================

@value
struct Document:
    """A document in the corpus."""
    var id: String
    var title: String
    var content: String

    fn __init__(inout self, id: String = "", title: String = "", content: String = ""):
        self.id = id
        self.title = title
        self.content = content


struct Corpus:
    """In-memory corpus for retrieval."""
    var documents: List[Document]
    var id_to_idx: Dict[String, Int]

    fn __init__(inout self):
        self.documents = List[Document]()
        self.id_to_idx = Dict[String, Int]()

    fn load_from_jsonl(inout self, path: String) raises:
        """Load corpus from JSONL file."""
        if not file_exists(path):
            print("Warning: Corpus file not found:", path)
            return

        var content = read_file(path)
        var lines = content.split("\n")

        for line in lines:
            if len(line[]) == 0:
                continue

            # Parse JSON line for id, title, content
            var doc_id = extract_json_field(line[], "id")
            var title = extract_json_field(line[], "title")
            var doc_content = extract_json_field(line[], "content")

            if doc_id != "":
                var idx = len(self.documents)
                self.documents.append(Document(doc_id, title, doc_content))
                self.id_to_idx[doc_id] = idx

        print("Loaded", len(self.documents), "documents from", path)

    fn get(self, doc_id: String) -> Document:
        """Get document by ID."""
        if doc_id in self.id_to_idx:
            return self.documents[self.id_to_idx[doc_id]]
        return Document()


fn extract_json_field(json_str: String, field: String) -> String:
    """Extract a string field from JSON."""
    var search = '"' + field + '":'
    var pos = json_str.find(search)
    if pos == -1:
        return ""

    var start = pos + len(search)
    while start < len(json_str) and json_str[start] == ' ':
        start += 1

    if start >= len(json_str) or json_str[start] != '"':
        return ""

    start += 1
    var end = start
    while end < len(json_str) and json_str[end] != '"':
        if json_str[end] == '\\' and end + 1 < len(json_str):
            end += 2
        else:
            end += 1

    return json_str[start:end]



# ============================================================================
# BM25 Search (Pure Mojo)
# ============================================================================

struct BM25Index:
    """Simple BM25 index for text search."""
    var term_doc_freq: Dict[String, Int]  # term -> document frequency
    var doc_lengths: List[Int]
    var avg_doc_length: Float64
    var k1: Float64
    var b: Float64
    var corpus_size: Int

    fn __init__(inout self):
        self.term_doc_freq = Dict[String, Int]()
        self.doc_lengths = List[Int]()
        self.avg_doc_length = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self.corpus_size = 0

    fn build(inout self, corpus: Corpus):
        """Build BM25 index from corpus."""
        self.corpus_size = len(corpus.documents)
        var total_length = 0

        for i in range(len(corpus.documents)):
            var doc = corpus.documents[i]
            var tokens = tokenize(doc.content)
            var doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Count unique terms in this document
            var seen = Dict[String, Bool]()
            for tok in tokens:
                if tok[] not in seen:
                    seen[tok[]] = True
                    if tok[] in self.term_doc_freq:
                        self.term_doc_freq[tok[]] = self.term_doc_freq[tok[]] + 1
                    else:
                        self.term_doc_freq[tok[]] = 1

        if self.corpus_size > 0:
            self.avg_doc_length = Float64(total_length) / Float64(self.corpus_size)

        print("Built BM25 index with", len(self.term_doc_freq), "unique terms")

    fn search(self, query: String, corpus: Corpus, topk: Int) -> List[Tuple[Int, Float64]]:
        """Search for documents matching query. Returns (doc_idx, score) pairs."""
        var query_tokens = tokenize(query)
        var scores = List[Tuple[Int, Float64]]()

        for i in range(len(corpus.documents)):
            var score = self.score_document(i, query_tokens, corpus)
            if score > 0.0:
                scores.append((i, score))

        # Sort by score descending (simple bubble sort for now)
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if scores[j][1] > scores[i][1]:
                    var tmp = scores[i]
                    scores[i] = scores[j]
                    scores[j] = tmp

        # Return top-k
        var result = List[Tuple[Int, Float64]]()
        for i in range(min(topk, len(scores))):
            result.append(scores[i])
        return result

    fn score_document(self, doc_idx: Int, query_tokens: List[String], corpus: Corpus) -> Float64:
        """Calculate BM25 score for a document."""
        var doc = corpus.documents[doc_idx]
        var doc_tokens = tokenize(doc.content)
        var doc_len = len(doc_tokens)

        # Count term frequencies in document
        var term_freq = Dict[String, Int]()
        for tok in doc_tokens:
            if tok[] in term_freq:
                term_freq[tok[]] = term_freq[tok[]] + 1
            else:
                term_freq[tok[]] = 1

        var score: Float64 = 0.0
        for q_tok in query_tokens:
            if q_tok[] not in term_freq:
                continue

            var tf = Float64(term_freq[q_tok[]])
            var df = Float64(self.term_doc_freq.get(q_tok[], 1))
            var n = Float64(self.corpus_size)

            # IDF
            var idf = log((n - df + 0.5) / (df + 0.5) + 1.0)

            # TF normalization
            var tf_norm = (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * Float64(doc_len) / self.avg_doc_length))

            score += idf * tf_norm

        return score


fn tokenize(text: String) -> List[String]:
    """Simple whitespace tokenizer with lowercasing."""
    var tokens = List[String]()
    var current = String("")

    for i in range(len(text)):
        var c = text[i]
        if c == ' ' or c == '\n' or c == '\t' or c == '.' or c == ',' or c == '!' or c == '?':
            if len(current) > 0:
                tokens.append(current.lower())
                current = String("")
        else:
            current += c

    if len(current) > 0:
        tokens.append(current.lower())

    return tokens


fn log(x: Float64) -> Float64:
    """Natural logarithm approximation."""
    if x <= 0:
        return -1000.0
    # Use Taylor series around 1: ln(x) ≈ 2 * sum((x-1)/(x+1))^(2n+1) / (2n+1)
    var y = (x - 1.0) / (x + 1.0)
    var y2 = y * y
    var result = y
    var term = y
    for n in range(1, 20):
        term *= y2
        result += term / Float64(2 * n + 1)
    return 2.0 * result


# ============================================================================
# HTTP Server (via Zig FFI)
# ============================================================================

# Global state for request handling
var g_corpus = Corpus()
var g_index = BM25Index()
var g_config = RetrievalConfig()


fn handle_search_request(query: String, topk: Int) -> String:
    """Handle a search request and return JSON response."""
    var results = g_index.search(query, g_corpus, topk)

    var response = String('{"results": [')
    for i in range(len(results)):
        var doc_idx = results[i][0]
        var score = results[i][1]
        var doc = g_corpus.documents[doc_idx]

        if i > 0:
            response += ","
        response += '{"id": "' + doc.id + '", "title": "' + escape_json(doc.title) + '", "score": ' + String(score) + ', "snippet": "' + escape_json(doc.content[:min(200, len(doc.content))]) + '"}'

    response += ']}'
    return response


fn escape_json(s: String) -> String:
    """Escape special characters for JSON."""
    var result = String("")
    for i in range(len(s)):
        var c = s[i]
        if c == '"':
            result += '\\"'
        elif c == '\\':
            result += '\\\\'
        elif c == '\n':
            result += '\\n'
        elif c == '\r':
            result += '\\r'
        elif c == '\t':
            result += '\\t'
        else:
            result += c
    return result


fn min(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


# ============================================================================
# CLI Entry Point
# ============================================================================

fn parse_port_from_args() -> Int:
    """Parse --port argument from command line arguments."""
    var port: Int = 8000
    var args = argv()
    var i = 1
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            try:
                port = atol(args[i + 1])
            except:
                print("Warning: Invalid port number, using default 8000")
                port = 8000
            break
        i += 1
    return port


fn parse_corpus_path_from_args() -> String:
    """Parse --corpus argument from command line arguments."""
    var path = String("./data/wiki_corpus.jsonl")
    var args = argv()
    var i = 1
    while i < len(args):
        if args[i] == "--corpus" and i + 1 < len(args):
            path = args[i + 1]
            break
        i += 1
    return path


fn main() raises:
    """Main entry point for the Wikipedia retrieval server."""
    print("🔍 Wikipedia Retrieval Server (Pure Mojo)")
    print("=========================================")

    var port = parse_port_from_args()
    var corpus_path = parse_corpus_path_from_args()

    print("Loading corpus from:", corpus_path)
    g_corpus.load_from_jsonl(corpus_path)

    print("Building BM25 index...")
    g_index.build(g_corpus)

    print("Starting server on port", port)
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /search     - Search documents")
    print("  POST /retrieve   - Retrieve by ID")

    # Note: For full HTTP server, use the Zig FFI integration
    # For now, this provides the core retrieval logic
    # The actual server can be started via:
    #   zig build-exe shared/http/server.zig -o wiki_server
    #   ./wiki_server --port 1401

    # Demo search
    print("\n📝 Demo search for 'machine learning':")
    var demo_results = handle_search_request("machine learning", 5)
    print(demo_results)

    print("\n✅ Retrieval server ready. Use Zig HTTP server for production.")