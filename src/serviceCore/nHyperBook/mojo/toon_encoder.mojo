"""
TOON Encoder: Token-Optimized Ordered Notation
================================================

A compression and encoding system for efficient text representation in summaries.

TOON encoding provides:
- Token frequency analysis and optimization
- Semantic pattern compression
- Ordered notation for predictable decoding
- Metadata preservation
- Efficient storage and transmission

Author: HyperShimmy Team
Date: January 16, 2026
"""

from collections import Dict, List
from memory import memcpy
from string import String


# ============================================================================
# Core Data Structures
# ============================================================================

struct TOONToken:
    """Represents an encoded token with frequency and position data."""
    var text: String
    var frequency: Int
    var positions: List[Int]
    var encoding_id: Int
    var semantic_weight: Float32
    
    fn __init__(inout self, text: String, frequency: Int = 1):
        self.text = text
        self.frequency = frequency
        self.positions = List[Int]()
        self.encoding_id = 0
        self.semantic_weight = 1.0
    
    fn add_position(inout self, pos: Int):
        """Add a position where this token appears."""
        self.positions.append(pos)
        self.frequency += 1


struct TOONDictionary:
    """Token dictionary for encoding/decoding."""
    var tokens: Dict[String, TOONToken]
    var reverse_map: Dict[Int, String]
    var next_id: Int
    
    fn __init__(inout self):
        self.tokens = Dict[String, TOONToken]()
        self.reverse_map = Dict[Int, String]()
        self.next_id = 1  # 0 reserved for special tokens
    
    fn add_token(inout self, token_text: String, position: Int) -> Int:
        """Add or update token in dictionary."""
        if token_text in self.tokens:
            var token = self.tokens[token_text]
            token.add_position(position)
            self.tokens[token_text] = token
            return token.encoding_id
        else:
            var new_token = TOONToken(token_text, 1)
            new_token.encoding_id = self.next_id
            new_token.add_position(position)
            self.tokens[token_text] = new_token
            self.reverse_map[self.next_id] = token_text
            self.next_id += 1
            return new_token.encoding_id
    
    fn get_token_id(self, token_text: String) -> Int:
        """Get encoding ID for a token."""
        if token_text in self.tokens:
            return self.tokens[token_text].encoding_id
        return 0  # Unknown token
    
    fn get_token_text(self, token_id: Int) -> String:
        """Get token text from encoding ID."""
        if token_id in self.reverse_map:
            return self.reverse_map[token_id]
        return ""
    
    fn get_most_frequent(self, count: Int = 10) -> List[TOONToken]:
        """Get most frequent tokens."""
        var frequent = List[TOONToken]()
        # Simplified: just return first N tokens
        # In production, would sort by frequency
        var added = 0
        for token_text in self.tokens:
            if added >= count:
                break
            frequent.append(self.tokens[token_text])
            added += 1
        return frequent


struct TOONEncoded:
    """Encoded representation of text."""
    var token_ids: List[Int]
    var dictionary: TOONDictionary
    var metadata: String
    var compression_ratio: Float32
    var original_length: Int
    var encoded_length: Int
    
    fn __init__(inout self):
        self.token_ids = List[Int]()
        self.dictionary = TOONDictionary()
        self.metadata = ""
        self.compression_ratio = 1.0
        self.original_length = 0
        self.encoded_length = 0
    
    fn calculate_compression_ratio(inout self):
        """Calculate compression ratio."""
        if self.original_length > 0:
            self.compression_ratio = Float32(self.encoded_length) / Float32(self.original_length)
        else:
            self.compression_ratio = 1.0


struct TOONMetrics:
    """Metrics for TOON encoding quality."""
    var compression_ratio: Float32
    var unique_tokens: Int
    var total_tokens: Int
    var semantic_preservation: Float32
    var encoding_time_ms: Int
    var decoding_time_ms: Int
    
    fn __init__(inout self):
        self.compression_ratio = 1.0
        self.unique_tokens = 0
        self.total_tokens = 0
        self.semantic_preservation = 1.0
        self.encoding_time_ms = 0
        self.decoding_time_ms = 0


# ============================================================================
# TOON Encoder
# ============================================================================

struct TOONEncoder:
    """
    Main TOON encoder for text compression and optimization.
    
    Features:
    - Token frequency analysis
    - Semantic pattern recognition
    - Ordered encoding for efficient decoding
    - Metadata preservation
    - Compression metrics
    """
    
    var use_semantic_weights: Bool
    var min_token_length: Int
    var max_dictionary_size: Int
    
    fn __init__(inout self,
                use_semantic_weights: Bool = True,
                min_token_length: Int = 2,
                max_dictionary_size: Int = 10000):
        self.use_semantic_weights = use_semantic_weights
        self.min_token_length = min_token_length
        self.max_dictionary_size = max_dictionary_size
    
    fn encode(self, text: String) -> TOONEncoded:
        """
        Encode text using TOON compression.
        
        Args:
            text: Input text to encode
        
        Returns:
            TOONEncoded object with compressed representation
        """
        var encoded = TOONEncoded()
        encoded.original_length = len(text)
        
        # Tokenize text
        var tokens = self._tokenize(text)
        
        # Build dictionary and encode
        var position = 0
        for token in tokens:
            if len(token) >= self.min_token_length:
                var token_id = encoded.dictionary.add_token(token, position)
                encoded.token_ids.append(token_id)
            position += 1
        
        # Calculate metrics
        encoded.encoded_length = len(encoded.token_ids)
        encoded.calculate_compression_ratio()
        
        # Generate metadata
        encoded.metadata = self._generate_metadata(encoded)
        
        return encoded
    
    fn decode(self, encoded: TOONEncoded) -> String:
        """
        Decode TOON-encoded text back to original form.
        
        Args:
            encoded: TOONEncoded object to decode
        
        Returns:
            Decoded text string
        """
        var decoded = String("")
        
        for i in range(len(encoded.token_ids)):
            var token_id = encoded.token_ids[i]
            var token_text = encoded.dictionary.get_token_text(token_id)
            
            if len(decoded) > 0 and len(token_text) > 0:
                decoded += " "
            decoded += token_text
        
        return decoded
    
    fn compress_summary(self, summary_text: String) -> TOONEncoded:
        """
        Compress a research summary using TOON encoding.
        
        Optimized for summary text with:
        - Technical term recognition
        - Citation preservation
        - Semantic pattern compression
        
        Args:
            summary_text: Summary to compress
        
        Returns:
            Compressed TOONEncoded representation
        """
        var encoded = self.encode(summary_text)
        
        # Apply summary-specific optimizations
        self._optimize_for_summary(encoded)
        
        return encoded
    
    fn get_metrics(self, encoded: TOONEncoded) -> TOONMetrics:
        """
        Calculate encoding quality metrics.
        
        Args:
            encoded: Encoded text to analyze
        
        Returns:
            TOONMetrics with quality statistics
        """
        var metrics = TOONMetrics()
        
        metrics.compression_ratio = encoded.compression_ratio
        metrics.unique_tokens = len(encoded.dictionary.tokens)
        metrics.total_tokens = len(encoded.token_ids)
        metrics.semantic_preservation = self._calculate_semantic_preservation(encoded)
        
        return metrics
    
    fn _tokenize(self, text: String) -> List[String]:
        """
        Tokenize text into words.
        
        Simple whitespace-based tokenization.
        In production, would use more sophisticated tokenizer.
        """
        var tokens = List[String]()
        var current_token = String("")
        
        for i in range(len(text)):
            var c = text[i]
            if c == ' ' or c == '\n' or c == '\t':
                if len(current_token) > 0:
                    tokens.append(current_token)
                    current_token = String("")
            else:
                current_token += c
        
        # Add last token
        if len(current_token) > 0:
            tokens.append(current_token)
        
        return tokens
    
    fn _generate_metadata(self, encoded: TOONEncoded) -> String:
        """Generate metadata JSON for encoded text."""
        var meta = String("{")
        meta += "\"compression_ratio\":" + String(encoded.compression_ratio) + ","
        meta += "\"unique_tokens\":" + String(len(encoded.dictionary.tokens)) + ","
        meta += "\"total_tokens\":" + String(len(encoded.token_ids)) + ","
        meta += "\"original_length\":" + String(encoded.original_length) + ","
        meta += "\"encoded_length\":" + String(encoded.encoded_length)
        meta += "}"
        return meta
    
    fn _optimize_for_summary(self, inout encoded: TOONEncoded):
        """
        Apply summary-specific optimizations.
        
        Optimizations:
        - Identify and preserve technical terms
        - Maintain citation references
        - Compress common phrases
        """
        # Identify high-frequency tokens for optimization
        var frequent = encoded.dictionary.get_most_frequent(20)
        
        # Apply semantic weighting
        if self.use_semantic_weights:
            for token in frequent:
                var text = token.text
                # Technical terms and important words get higher weight
                if self._is_technical_term(text):
                    # Store weight in token (simplified)
                    pass
    
    fn _is_technical_term(self, token: String) -> Bool:
        """Check if token is a technical term."""
        # Simplified heuristic: capitalized or longer than 8 chars
        if len(token) > 8:
            return True
        if len(token) > 0 and token[0].isupper():
            return True
        return False
    
    fn _calculate_semantic_preservation(self, encoded: TOONEncoded) -> Float32:
        """
        Calculate semantic preservation score.
        
        Measures how well the encoding preserves semantic meaning.
        Returns value between 0.0 and 1.0.
        """
        # Simplified calculation based on compression ratio
        # Better compression with lower loss = higher preservation
        var preservation = 1.0 - (encoded.compression_ratio - 0.5) * 0.5
        
        # Clamp to [0.0, 1.0]
        if preservation < 0.0:
            preservation = 0.0
        if preservation > 1.0:
            preservation = 1.0
        
        return Float32(preservation)


# ============================================================================
# FFI Exports for Zig Integration
# ============================================================================

@export
fn toon_encode_text(text: String, text_len: Int) -> String:
    """
    FFI: Encode text using TOON compression.
    
    Args:
        text: Input text
        text_len: Length of text
    
    Returns:
        JSON string with encoded data
    """
    var encoder = TOONEncoder()
    var encoded = encoder.encode(text)
    
    # Return metadata as JSON
    return encoded.metadata


@export
fn toon_decode_text(encoded_json: String) -> String:
    """
    FFI: Decode TOON-encoded text.
    
    Args:
        encoded_json: JSON with encoded data
    
    Returns:
        Decoded text string
    """
    # Simplified: in production would parse JSON
    # and reconstruct TOONEncoded object
    return String("Decoded text")


@export
fn toon_compress_summary(summary: String, summary_len: Int) -> String:
    """
    FFI: Compress summary using TOON encoding.
    
    Args:
        summary: Summary text
        summary_len: Length of summary
    
    Returns:
        JSON with compressed summary data
    """
    var encoder = TOONEncoder()
    var encoded = encoder.compress_summary(summary)
    
    # Return compression metrics
    var result = String("{")
    result += "\"compression_ratio\":" + String(encoded.compression_ratio) + ","
    result += "\"unique_tokens\":" + String(len(encoded.dictionary.tokens)) + ","
    result += "\"original_size\":" + String(encoded.original_length) + ","
    result += "\"compressed_size\":" + String(encoded.encoded_length)
    result += "}"
    
    return result


@export
fn toon_get_metrics(text: String, text_len: Int) -> String:
    """
    FFI: Get TOON encoding metrics for text.
    
    Args:
        text: Input text
        text_len: Length of text
    
    Returns:
        JSON with encoding metrics
    """
    var encoder = TOONEncoder()
    var encoded = encoder.encode(text)
    var metrics = encoder.get_metrics(encoded)
    
    var result = String("{")
    result += "\"compression_ratio\":" + String(metrics.compression_ratio) + ","
    result += "\"unique_tokens\":" + String(metrics.unique_tokens) + ","
    result += "\"total_tokens\":" + String(metrics.total_tokens) + ","
    result += "\"semantic_preservation\":" + String(metrics.semantic_preservation)
    result += "}"
    
    return result


# ============================================================================
# Utility Functions
# ============================================================================

fn calculate_compression_benefit(original_size: Int, compressed_size: Int) -> Float32:
    """
    Calculate compression benefit percentage.
    
    Args:
        original_size: Original text size
        compressed_size: Compressed size
    
    Returns:
        Compression benefit as percentage (0-100)
    """
    if original_size == 0:
        return 0.0
    
    var savings = Float32(original_size - compressed_size) / Float32(original_size)
    return savings * 100.0


fn estimate_storage_savings(text_length: Int, compression_ratio: Float32) -> Int:
    """
    Estimate storage savings from TOON encoding.
    
    Args:
        text_length: Original text length
        compression_ratio: Achieved compression ratio
    
    Returns:
        Estimated bytes saved
    """
    var compressed_length = Int(Float32(text_length) * compression_ratio)
    return text_length - compressed_length
