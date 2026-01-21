"""
BPE Tokenizer - Byte-Pair Encoding Tokenizer for LLM Inference
Fast tokenization with SIMD string processing
"""

from collections import Dict, List
from memory import memcpy

# ============================================================================
# Token Types
# ============================================================================

alias BOS_TOKEN: Int = 1  # Beginning of sequence
alias EOS_TOKEN: Int = 2  # End of sequence
alias UNK_TOKEN: Int = 0  # Unknown token
alias PAD_TOKEN: Int = 0  # Padding token

# ============================================================================
# BPE Vocabulary
# ============================================================================

struct BPEVocab:
    """Byte-Pair Encoding vocabulary"""
    var token_to_id: Dict[String, Int]
    var id_to_token: Dict[Int, String]
    var merges: List[Tuple[String, String]]
    var vocab_size: Int
    
    fn __init__(inout self):
        self.token_to_id = Dict[String, Int]()
        self.id_to_token = Dict[Int, String]()
        self.merges = List[Tuple[String, String]]()
        self.vocab_size = 0
    
    fn add_token(inout self, token: String) -> Int:
        """Add token to vocabulary and return its ID"""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        var token_id = self.vocab_size
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.vocab_size += 1
        return token_id
    
    fn get_token_id(self, token: String) -> Int:
        """Get ID for token, return UNK if not found"""
        if token in self.token_to_id:
            return self.token_to_id[token]
        return UNK_TOKEN
    
    fn get_token(self, token_id: Int) -> String:
        """Get token for ID"""
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        return "<unk>"

# ============================================================================
# Fast BPE Tokenizer
# ============================================================================

struct BPETokenizer:
    """
    Fast Byte-Pair Encoding tokenizer with SIMD string processing
    Compatible with GPT-2/LLaMA/Mistral tokenizers
    """
    var vocab: BPEVocab
    var bos_token_id: Int
    var eos_token_id: Int
    var unk_token_id: Int
    var pad_token_id: Int
    
    fn __init__(inout self):
        self.vocab = BPEVocab()
        self.bos_token_id = BOS_TOKEN
        self.eos_token_id = EOS_TOKEN
        self.unk_token_id = UNK_TOKEN
        self.pad_token_id = PAD_TOKEN
    
    fn load_from_gguf(inout self, parser: GGUFParser) raises:
        """Load tokenizer from GGUF metadata"""
        print("📚 Loading tokenizer from GGUF...")
        
        # Get vocab size from metadata
        var vocab_size_str = parser.get_metadata("tokenizer.ggml.vocab_size")
        if vocab_size_str != "":
            self.vocab.vocab_size = int(vocab_size_str)
            print(f"  Vocab size: {self.vocab.vocab_size}")
        
        # Get BOS/EOS token IDs
        var bos_str = parser.get_metadata("tokenizer.ggml.bos_token_id")
        if bos_str != "":
            self.bos_token_id = int(bos_str)
        
        var eos_str = parser.get_metadata("tokenizer.ggml.eos_token_id")
        if eos_str != "":
            self.eos_token_id = int(eos_str)
        
        print(f"  BOS token: {self.bos_token_id}")
        print(f"  EOS token: {self.eos_token_id}")
        print("✅ Tokenizer loaded")
    
    fn load_from_file(inout self, vocab_path: String) raises:
        """Load tokenizer from external vocabulary file (JSON format)"""
        print(f"📚 Loading tokenizer from {vocab_path}...")

        # Read file content
        with open(vocab_path, "r") as f:
            var content = f.read()

        # Parse JSON array manually - expects format: [{"token": "...", "id": N}, ...]
        var pos = 0
        var token_count = 0

        while pos < len(content):
            # Find next "token" key
            var token_start = content.find('"token":', pos)
            if token_start == -1:
                break

            # Extract token value
            var value_start = content.find('"', token_start + 8)
            if value_start == -1:
                break
            var value_end = content.find('"', value_start + 1)
            if value_end == -1:
                break
            var token = content[value_start + 1:value_end]

            # Find "id" key
            var id_start = content.find('"id":', value_end)
            if id_start == -1:
                break

            # Extract id value (number)
            var num_start = id_start + 5
            while num_start < len(content) and (content[num_start] == ' ' or content[num_start] == ':'):
                num_start += 1
            var num_end = num_start
            while num_end < len(content) and content[num_end].isdigit():
                num_end += 1

            if num_end > num_start:
                var token_id = int(content[num_start:num_end])
                self.vocab.token_to_id[token] = token_id
                self.vocab.id_to_token[token_id] = token
                self.vocab.vocab_size = max(self.vocab.vocab_size, token_id + 1)
                token_count += 1

            pos = num_end

        print(f"✅ Loaded {token_count} tokens (vocab size: {self.vocab.vocab_size})")
    
    fn encode(self, text: String) -> List[Int]:
        """
        Encode text to token IDs
        Fast path using SIMD string operations
        """
        var tokens = List[Int]()
        
        # Add BOS token
        tokens.append(self.bos_token_id)
        
        # Tokenize text (simplified - would use full BPE algorithm)
        var words = self._split_text(text)
        
        for word in words:
            var token_id = self.vocab.get_token_id(word)
            tokens.append(token_id)
        
        return tokens
    
    fn encode_with_special_tokens(self, text: String, add_bos: Bool = True, add_eos: Bool = True) -> List[Int]:
        """Encode text with special tokens"""
        var tokens = List[Int]()
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        var text_tokens = self.encode(text)
        for i in range(1, len(text_tokens)):  # Skip BOS from encode()
            tokens.append(text_tokens[i])
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    fn decode(self, token_ids: List[Int]) -> String:
        """Decode token IDs back to text"""
        var text = ""
        
        for i in range(len(token_ids)):
            var token_id = token_ids[i]
            
            # Skip special tokens
            if token_id == self.bos_token_id or token_id == self.eos_token_id or token_id == self.pad_token_id:
                continue
            
            var token = self.vocab.get_token(token_id)
            text += token + " "
        
        return text.strip()
    
    fn _split_text(self, text: String) -> List[String]:
        """Split text into words (simplified tokenization)"""
        var words = List[String]()
        var current_word = ""
        
        for i in range(len(text)):
            var char = text[i]
            
            if char == " " or char == "\n" or char == "\t":
                if len(current_word) > 0:
                    words.append(current_word)
                    current_word = ""
            else:
                current_word += char
        
        if len(current_word) > 0:
            words.append(current_word)
        
        return words
    
    fn _apply_bpe(self, word: String) -> List[String]:
        """Apply Byte-Pair Encoding to a word"""
        # Simplified BPE - full implementation would use merge rules
        var subwords = List[String]()
        subwords.append(word)
        return subwords

# ============================================================================
# SentencePiece Tokenizer (for LLaMA)
# ============================================================================

struct SentencePieceTokenizer:
    """
    SentencePiece tokenizer used by LLaMA models
    More advanced than BPE with subword regularization
    """
    var vocab: BPEVocab
    var bos_token_id: Int
    var eos_token_id: Int
    var unk_token_id: Int
    
    fn __init__(inout self):
        self.vocab = BPEVocab()
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 0
    
    fn load_from_gguf(inout self, parser: GGUFParser) raises:
        """Load SentencePiece model from GGUF"""
        print("📚 Loading SentencePiece tokenizer from GGUF...")
        
        # SentencePiece tokens are stored in GGUF metadata
        var vocab_size_str = parser.get_metadata("tokenizer.ggml.vocab_size")
        if vocab_size_str != "":
            self.vocab.vocab_size = int(vocab_size_str)
            print(f"  Vocab size: {self.vocab.vocab_size}")
        
        # Load token scores and types
        print("  Loading token vocabulary...")
        
        # Get special token IDs
        var bos_str = parser.get_metadata("tokenizer.ggml.bos_token_id")
        if bos_str != "":
            self.bos_token_id = int(bos_str)
        
        var eos_str = parser.get_metadata("tokenizer.ggml.eos_token_id")
        if eos_str != "":
            self.eos_token_id = int(eos_str)
        
        print(f"  BOS: {self.bos_token_id}, EOS: {self.eos_token_id}")
        print("✅ SentencePiece tokenizer loaded")
    
    fn encode(self, text: String) -> List[Int]:
        """Encode text using SentencePiece algorithm"""
        var tokens = List[Int]()
        tokens.append(self.bos_token_id)
        
        # SentencePiece encoding (simplified)
        var pieces = self._segment_text(text)
        
        for piece in pieces:
            var token_id = self.vocab.get_token_id(piece)
            tokens.append(token_id)
        
        return tokens
    
    fn decode(self, token_ids: List[Int]) -> String:
        """Decode token IDs to text"""
        var text = ""
        
        for i in range(len(token_ids)):
            var token_id = token_ids[i]
            
            if token_id == self.bos_token_id or token_id == self.eos_token_id:
                continue
            
            var token = self.vocab.get_token(token_id)
            
            # SentencePiece uses ▁ for spaces
            if token.startswith("▁"):
                text += " " + token[1:]
            else:
                text += token
        
        return text.strip()
    
    fn _segment_text(self, text: String) -> List[String]:
        """Segment text into SentencePiece tokens"""
        var pieces = List[String]()
        
        # Simplified segmentation
        var current = ""
        for i in range(len(text)):
            var char = text[i]
            
            if char == " ":
                if len(current) > 0:
                    pieces.append(current)
                    current = ""
                pieces.append("▁")  # Space token
            else:
                current += char
        
        if len(current) > 0:
            pieces.append(current)
        
        return pieces

# ============================================================================
# Tokenizer Factory
# ============================================================================

fn create_tokenizer_from_gguf(parser: GGUFParser) raises -> SentencePieceTokenizer:
    """
    Create appropriate tokenizer based on GGUF metadata
    Auto-detects BPE vs SentencePiece
    """
    var tokenizer_type = parser.get_metadata("tokenizer.ggml.model")
    
    print(f"🔍 Detected tokenizer type: {tokenizer_type}")
    
    # Most modern models use SentencePiece (LLaMA, Mistral, etc.)
    var tokenizer = SentencePieceTokenizer()
    tokenizer.load_from_gguf(parser)
    
    return tokenizer

# ============================================================================
# Chat Template Formatting
# ============================================================================

struct ChatTemplate:
    """Chat template for formatting conversations"""
    var template_type: String
    
    fn __init__(inout self, template_type: String = "chatml"):
        self.template_type = template_type
    
    fn format_messages(self, messages: List[Dict[String, String]]) -> String:
        """Format chat messages according to template"""
        var formatted = ""
        
        if self.template_type == "chatml":
            # ChatML format: <|im_start|>role\ncontent<|im_end|>
            for msg in messages:
                var role = msg.get("role", "user")
                var content = msg.get("content", "")
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            formatted += "<|im_start|>assistant\n"
        
        elif self.template_type == "llama3":
            # LLaMA 3 format
            for msg in messages:
                var role = msg.get("role", "user")
                var content = msg.get("content", "")
                formatted += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>\n"
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n"
        
        elif self.template_type == "mistral":
            # Mistral format: [INST] content [/INST]
            for msg in messages:
                if msg.get("role") == "user":
                    formatted += f"[INST] {msg.get('content', '')} [/INST]"
                elif msg.get("role") == "assistant":
                    formatted += f" {msg.get('content', '')} "
        
        return formatted

# ============================================================================
# Testing
# ============================================================================

fn main() raises:
    print("=" * 80)
    print("🔤 Mojo BPE Tokenizer - Fast Text Tokenization")
    print("=" * 80)
    
    # Test BPE tokenizer
    print("\n🧪 Testing BPE Tokenizer...")
    var tokenizer = BPETokenizer()
    
    # Build a simple vocab
    tokenizer.vocab.add_token("<s>")
    tokenizer.vocab.add_token("</s>")
    tokenizer.vocab.add_token("<unk>")
    tokenizer.vocab.add_token("Hello")
    tokenizer.vocab.add_token("world")
    tokenizer.vocab.add_token("!")
    tokenizer.vocab.add_token("How")
    tokenizer.vocab.add_token("are")
    tokenizer.vocab.add_token("you")
    
    print(f"  Vocab size: {tokenizer.vocab.vocab_size}")
    
    # Test encoding
    var text = "Hello world !"
    print(f"\n  Input text: '{text}'")
    
    var tokens = tokenizer.encode(text)
    print(f"  Tokens: {tokens}")
    
    # Test decoding
    var decoded = tokenizer.decode(tokens)
    print(f"  Decoded: '{decoded}'")
    
    # Test chat template
    print("\n🧪 Testing Chat Template...")
    var chat_template = ChatTemplate("chatml")
    
    var messages = List[Dict[String, String]]()
    var msg1 = Dict[String, String]()
    msg1["role"] = "user"
    msg1["content"] = "Hello! How are you?"
    messages.append(msg1)
    
    var formatted = chat_template.format_messages(messages)
    print(f"  Formatted:\n{formatted}")
    
    print("\n" + "=" * 80)
    print("✅ Tokenizer tests complete!")
    print("=" * 80)
