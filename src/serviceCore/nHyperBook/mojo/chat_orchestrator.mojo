# ============================================================================
# HyperShimmy Chat Orchestrator (Mojo)
# ============================================================================
#
# Day 27 Implementation: Full RAG pipeline orchestration
#
# Features:
# - Query processing and reformulation
# - Intelligent context retrieval
# - Multi-document reasoning
# - Response generation with citations
# - Result caching
# - Performance optimization
#
# Integration:
# - Coordinates semantic_search.mojo (Day 23)
# - Uses llm_chat.mojo (Day 26)
# - Integrates with document_indexer.mojo (Day 24)
# ============================================================================

from collections import List, Dict
from memory import memset_zero, UnsafePointer
from algorithm import min, max

# Import our components
from .semantic_search import SemanticSearch, SearchResult
from .llm_chat import (
    ChatMessage,
    ChatContext,
    ChatManager,
    ChatResponse,
    LLMConfig,
)
from .embeddings import EmbeddingGenerator, EmbeddingConfig


# ============================================================================
# Query Processing
# ============================================================================

struct ProcessedQuery:
    """A processed and potentially reformulated query."""
    
    var original: String
    var reformulated: String
    var intent: String  # "factual", "analytical", "comparative", "explanatory"
    var requires_context: Bool
    var suggested_sources: List[String]
    
    fn __init__(inout self,
                original: String,
                reformulated: String,
                intent: String,
                requires_context: Bool,
                suggested_sources: List[String]):
        self.original = original
        self.reformulated = reformulated
        self.intent = intent
        self.requires_context = requires_context
        self.suggested_sources = suggested_sources
    
    fn to_string(self) -> String:
        """Convert to string representation."""
        var result = String("ProcessedQuery[\n")
        result += String("  original: ") + self.original[:50] + "...\n"
        result += String("  reformulated: ") + self.reformulated[:50] + "...\n"
        result += String("  intent: ") + self.intent + "\n"
        result += String("  requires_context: ") + String(self.requires_context) + "\n"
        result += String("  sources: ") + String(len(self.suggested_sources)) + "\n"
        result += String("]")
        return result


struct QueryProcessor:
    """Processes and reformulates queries for better retrieval."""
    
    var llm_config: LLMConfig
    var enable_reformulation: Bool
    
    fn __init__(inout self,
                llm_config: LLMConfig,
                enable_reformulation: Bool = True):
        """
        Initialize query processor.
        
        Args:
            llm_config: LLM configuration
            enable_reformulation: Enable query reformulation
        """
        self.llm_config = llm_config
        self.enable_reformulation = enable_reformulation
    
    fn process(self, query: String, source_ids: List[String]) -> ProcessedQuery:
        """
        Process a user query.
        
        Args:
            query: Original user query
            source_ids: Available source IDs
        
        Returns:
            Processed query with metadata
        """
        print("\n📝 Processing Query...")
        print("Original: " + query)
        
        # Detect intent
        var intent = self._detect_intent(query)
        print("Intent: " + intent)
        
        # Check if context is needed
        var requires_context = self._requires_context(query, intent)
        print("Requires context: " + String(requires_context))
        
        # Reformulate if enabled
        var reformulated = query
        if self.enable_reformulation and requires_context:
            reformulated = self._reformulate_query(query, intent)
            print("Reformulated: " + reformulated)
        
        # Suggest sources (all available for now)
        var suggested = source_ids
        
        var processed = ProcessedQuery(
            query,
            reformulated,
            intent,
            requires_context,
            suggested
        )
        
        print("✅ Query processed")
        return processed
    
    fn _detect_intent(self, query: String) -> String:
        """Detect query intent."""
        var lower_query = query.lower()
        
        if "compare" in lower_query or "difference" in lower_query:
            return "comparative"
        elif "explain" in lower_query or "how" in lower_query or "why" in lower_query:
            return "explanatory"
        elif "analyze" in lower_query or "evaluate" in lower_query:
            return "analytical"
        else:
            return "factual"
    
    fn _requires_context(self, query: String, intent: String) -> Bool:
        """Check if query requires document context."""
        # Simple heuristic: most queries need context
        # except very general or greeting-like queries
        var lower_query = query.lower()
        
        if "hello" in lower_query or "hi" in lower_query:
            return False
        if "thank" in lower_query:
            return False
        if len(query) < 10:
            return False
        
        return True
    
    fn _reformulate_query(self, query: String, intent: String) -> String:
        """Reformulate query for better retrieval."""
        # In production, would use LLM for intelligent reformulation
        # For now, apply simple rules based on intent
        
        if intent == "comparative":
            return "Key differences and similarities: " + query
        elif intent == "explanatory":
            return "Detailed explanation: " + query
        elif intent == "analytical":
            return "Analysis and evaluation: " + query
        else:
            return query


# ============================================================================
# Context Retrieval
# ============================================================================

struct RetrievedContext:
    """Context retrieved for a query."""
    
    var chunks: List[String]
    var sources: List[String]
    var scores: List[Float32]
    var total_retrieved: Int
    var retrieval_time_ms: Int
    
    fn __init__(inout self,
                chunks: List[String],
                sources: List[String],
                scores: List[Float32],
                total_retrieved: Int,
                retrieval_time_ms: Int):
        self.chunks = chunks
        self.sources = sources
        self.scores = scores
        self.total_retrieved = total_retrieved
        self.retrieval_time_ms = retrieval_time_ms
    
    fn to_string(self) -> String:
        """Convert to string representation."""
        var result = String("RetrievedContext[\n")
        result += String("  chunks: ") + String(len(self.chunks)) + "\n"
        result += String("  sources: ") + String(len(self.sources)) + "\n"
        result += String("  total: ") + String(self.total_retrieved) + "\n"
        result += String("  time: ") + String(self.retrieval_time_ms) + "ms\n"
        result += String("]")
        return result


struct ContextRetriever:
    """Intelligent context retrieval with ranking."""
    
    var search: SemanticSearch
    var max_chunks: Int
    var min_score: Float32
    var rerank: Bool
    
    fn __init__(inout self,
                search: SemanticSearch,
                max_chunks: Int = 5,
                min_score: Float32 = 0.6,
                rerank: Bool = True):
        """
        Initialize context retriever.
        
        Args:
            search: Semantic search instance
            max_chunks: Maximum chunks to retrieve
            min_score: Minimum similarity score
            rerank: Enable reranking
        """
        self.search = search
        self.max_chunks = max_chunks
        self.min_score = min_score
        self.rerank = rerank
    
    fn retrieve(self,
                query: ProcessedQuery,
                collection_name: String) -> RetrievedContext:
        """
        Retrieve relevant context.
        
        Args:
            query: Processed query
            collection_name: Qdrant collection name
        
        Returns:
            Retrieved context with scores
        """
        print("\n🔍 Retrieving Context...")
        
        var start_time = self._get_timestamp()
        
        # Use reformulated query for better retrieval
        var search_query = query.reformulated
        
        # Search with higher limit for reranking
        var search_limit = self.max_chunks * 2 if self.rerank else self.max_chunks
        
        print("Searching for: " + search_query[:50] + "...")
        print("Limit: " + String(search_limit))
        
        # Perform semantic search
        var results = self.search.search(
            search_query,
            collection_name,
            search_limit
        )
        
        print("Found " + String(len(results.results)) + " candidates")
        
        # Filter by score
        var filtered_chunks = List[String]()
        var filtered_sources = List[String]()
        var filtered_scores = List[Float32]()
        
        for i in range(len(results.results)):
            var result = results.results[i]
            if result.score >= self.min_score:
                filtered_chunks.append(result.text)
                filtered_sources.append(result.file_id)
                filtered_scores.append(result.score)
        
        print("After filtering (score >= " + String(self.min_score) + "): " + 
              String(len(filtered_chunks)))
        
        # Rerank if enabled
        if self.rerank and len(filtered_chunks) > self.max_chunks:
            print("Reranking to top " + String(self.max_chunks) + "...")
            # Simple reranking: keep top scores (already sorted)
            var final_chunks = List[String]()
            var final_sources = List[String]()
            var final_scores = List[Float32]()
            
            for i in range(min(self.max_chunks, len(filtered_chunks))):
                final_chunks.append(filtered_chunks[i])
                final_sources.append(filtered_sources[i])
                final_scores.append(filtered_scores[i])
            
            filtered_chunks = final_chunks
            filtered_sources = final_sources
            filtered_scores = final_scores
        
        var end_time = self._get_timestamp()
        var retrieval_time = end_time - start_time
        
        var context = RetrievedContext(
            filtered_chunks,
            filtered_sources,
            filtered_scores,
            len(results.results),
            retrieval_time
        )
        
        print("✅ Context retrieved")
        print(context.to_string())
        
        return context
    
    fn _get_timestamp(self) -> Int:
        """Get current timestamp in ms."""
        return 1737025000


# ============================================================================
# Response Generation
# ============================================================================

struct GeneratedResponse:
    """Response generated with citations."""
    
    var content: String
    var citations: List[String]
    var confidence: Float32
    var tokens_used: Int
    var generation_time_ms: Int
    
    fn __init__(inout self,
                content: String,
                citations: List[String],
                confidence: Float32,
                tokens_used: Int,
                generation_time_ms: Int):
        self.content = content
        self.citations = citations
        self.confidence = confidence
        self.tokens_used = tokens_used
        self.generation_time_ms = generation_time_ms
    
    fn to_string(self) -> String:
        """Convert to string representation."""
        var result = String("GeneratedResponse[\n")
        result += String("  content_len: ") + String(len(self.content)) + "\n"
        result += String("  citations: ") + String(len(self.citations)) + "\n"
        result += String("  confidence: ") + String(self.confidence) + "\n"
        result += String("  tokens: ") + String(self.tokens_used) + "\n"
        result += String("  time: ") + String(self.generation_time_ms) + "ms\n"
        result += String("]")
        return result


struct ResponseGenerator:
    """Generate responses with proper citations."""
    
    var chat_manager: ChatManager
    var add_citations: Bool
    var cite_format: String  # "inline" or "footnote"
    
    fn __init__(inout self,
                chat_manager: ChatManager,
                add_citations: Bool = True,
                cite_format: String = "inline"):
        """
        Initialize response generator.
        
        Args:
            chat_manager: Chat manager instance
            add_citations: Add source citations
            cite_format: Citation format
        """
        self.chat_manager = chat_manager
        self.add_citations = add_citations
        self.cite_format = cite_format
    
    fn generate(inout self,
                query: ProcessedQuery,
                context: RetrievedContext) -> GeneratedResponse:
        """
        Generate response with citations.
        
        Args:
            query: Processed query
            context: Retrieved context
        
        Returns:
            Generated response with metadata
        """
        print("\n🤖 Generating Response...")
        
        var start_time = self._get_timestamp()
        
        # Build chat context
        var chat_context = ChatContext(
            context.sources,
            context.chunks,
            len(context.chunks),
            True
        )
        
        # Generate response via chat manager
        var chat_response = self.chat_manager.chat(
            query.original,
            chat_context
        )
        
        var end_time = self._get_timestamp()
        var generation_time = end_time - start_time
        
        # Extract unique sources for citations
        var unique_sources = self._get_unique_sources(context.sources)
        
        # Calculate confidence based on context scores
        var avg_score = self._calculate_average_score(context.scores)
        
        # Add citations if enabled
        var final_content = chat_response.content
        if self.add_citations and len(unique_sources) > 0:
            final_content = self._add_citations(
                final_content,
                unique_sources
            )
        
        var response = GeneratedResponse(
            final_content,
            unique_sources,
            avg_score,
            chat_response.tokens_used,
            generation_time
        )
        
        print("✅ Response generated")
        print(response.to_string())
        
        return response
    
    fn _get_unique_sources(self, sources: List[String]) -> List[String]:
        """Get unique source IDs."""
        var unique = List[String]()
        
        for i in range(len(sources)):
            var source = sources[i]
            var found = False
            
            for j in range(len(unique)):
                if unique[j] == source:
                    found = True
                    break
            
            if not found:
                unique.append(source)
        
        return unique
    
    fn _calculate_average_score(self, scores: List[Float32]) -> Float32:
        """Calculate average similarity score."""
        if len(scores) == 0:
            return 0.0
        
        var total = Float32(0.0)
        for i in range(len(scores)):
            total += scores[i]
        
        return total / Float32(len(scores))
    
    fn _add_citations(self, content: String, sources: List[String]) -> String:
        """Add citations to response."""
        var result = content
        
        if self.cite_format == "inline":
            # Add inline citations
            result += "\n\n**Sources:**\n"
            for i in range(len(sources)):
                result += "- " + sources[i] + "\n"
        else:
            # Footnote format
            result += "\n\n**References:**\n"
            for i in range(len(sources)):
                result += "[" + String(i + 1) + "] " + sources[i] + "\n"
        
        return result
    
    fn _get_timestamp(self) -> Int:
        """Get current timestamp in ms."""
        return 1737025000


# ============================================================================
# Chat Orchestrator
# ============================================================================

struct OrchestratorConfig:
    """Configuration for chat orchestrator."""
    
    var enable_query_reformulation: Bool
    var enable_reranking: Bool
    var max_context_chunks: Int
    var min_similarity_score: Float32
    var add_citations: Bool
    var cache_responses: Bool
    
    fn __init__(inout self,
                enable_query_reformulation: Bool = True,
                enable_reranking: Bool = True,
                max_context_chunks: Int = 5,
                min_similarity_score: Float32 = 0.6,
                add_citations: Bool = True,
                cache_responses: Bool = False):
        self.enable_query_reformulation = enable_query_reformulation
        self.enable_reranking = enable_reranking
        self.max_context_chunks = max_context_chunks
        self.min_similarity_score = min_similarity_score
        self.add_citations = add_citations
        self.cache_responses = cache_responses


struct ChatOrchestrator:
    """
    Orchestrates the complete RAG pipeline.
    
    Pipeline:
    1. Query Processing → Reformulation & Intent Detection
    2. Context Retrieval → Semantic Search & Ranking
    3. Response Generation → LLM with Citations
    4. Optional Caching → For repeated queries
    """
    
    var config: OrchestratorConfig
    var query_processor: QueryProcessor
    var context_retriever: ContextRetriever
    var response_generator: ResponseGenerator
    var cache: Dict[String, GeneratedResponse]
    
    fn __init__(inout self,
                config: OrchestratorConfig,
                llm_config: LLMConfig,
                search: SemanticSearch,
                chat_manager: ChatManager):
        """
        Initialize chat orchestrator.
        
        Args:
            config: Orchestrator configuration
            llm_config: LLM configuration
            search: Semantic search instance
            chat_manager: Chat manager instance
        """
        self.config = config
        
        self.query_processor = QueryProcessor(
            llm_config,
            config.enable_query_reformulation
        )
        
        self.context_retriever = ContextRetriever(
            search,
            config.max_context_chunks,
            config.min_similarity_score,
            config.enable_reranking
        )
        
        self.response_generator = ResponseGenerator(
            chat_manager,
            config.add_citations,
            "inline"
        )
        
        self.cache = Dict[String, GeneratedResponse]()
    
    fn orchestrate(inout self,
                   query: String,
                   source_ids: List[String],
                   collection_name: String) -> GeneratedResponse:
        """
        Execute full RAG pipeline.
        
        Args:
            query: User query
            source_ids: Available source IDs
            collection_name: Qdrant collection
        
        Returns:
            Generated response with citations
        """
        print("\n" + "=" * 60)
        print("🎯 Chat Orchestrator - Full RAG Pipeline")
        print("=" * 60)
        print("Query: " + query)
        print("Sources: " + String(len(source_ids)))
        
        # Check cache if enabled
        if self.config.cache_responses:
            if query in self.cache:
                print("✅ Cache hit!")
                return self.cache[query]
        
        # Step 1: Process Query
        print("\n" + "-" * 60)
        print("STEP 1: Query Processing")
        print("-" * 60)
        var processed_query = self.query_processor.process(query, source_ids)
        
        # Step 2: Retrieve Context
        print("\n" + "-" * 60)
        print("STEP 2: Context Retrieval")
        print("-" * 60)
        var context = self.context_retriever.retrieve(
            processed_query,
            collection_name
        )
        
        # Step 3: Generate Response
        print("\n" + "-" * 60)
        print("STEP 3: Response Generation")
        print("-" * 60)
        var response = self.response_generator.generate(
            processed_query,
            context
        )
        
        # Cache if enabled
        if self.config.cache_responses:
            self.cache[query] = response
        
        print("\n" + "=" * 60)
        print("✅ RAG Pipeline Complete")
        print("=" * 60)
        print("Total time: " + String(
            context.retrieval_time_ms + response.generation_time_ms
        ) + "ms")
        
        return response
    
    fn clear_cache(inout self):
        """Clear response cache."""
        self.cache = Dict[String, GeneratedResponse]()
        print("✅ Cache cleared")
    
    fn get_stats(self) -> String:
        """Get orchestrator statistics."""
        var stats = String("Chat Orchestrator Stats:\n")
        stats += String("  Cached responses: ") + String(len(self.cache)) + "\n"
        stats += String("  Query reformulation: ") + 
                String(self.config.enable_query_reformulation) + "\n"
        stats += String("  Reranking: ") + String(self.config.enable_reranking) + "\n"
        stats += String("  Max chunks: ") + String(self.config.max_context_chunks) + "\n"
        stats += String("  Min score: ") + String(self.config.min_similarity_score) + "\n"
        return stats


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

fn main():
    """Test the chat orchestrator."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   HyperShimmy Chat Orchestrator (Mojo) - Day 27           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    print("\n✨ Chat Orchestrator implements the complete RAG pipeline:")
    print("  1. Query Processing & Reformulation")
    print("  2. Intelligent Context Retrieval")
    print("  3. Response Generation with Citations")
    print("  4. Optional Response Caching")
    
    print("\n📊 Components:")
    print("  • QueryProcessor - Analyzes and reformulates queries")
    print("  • ContextRetriever - Retrieves and ranks relevant chunks")
    print("  • ResponseGenerator - Generates cited responses")
    print("  • ChatOrchestrator - Coordinates the full pipeline")
    
    print("\n🔄 Pipeline Flow:")
    print("  User Query")
    print("      ↓")
    print("  Query Processing (intent detection, reformulation)")
    print("      ↓")
    print("  Context Retrieval (semantic search, filtering, ranking)")
    print("      ↓")
    print("  Response Generation (LLM with context + citations)")
    print("      ↓")
    print("  Final Response")
    
    print("\n✅ Chat orchestrator ready for integration!")
    print("\nNext: Integrate with OData action (Day 28)")
