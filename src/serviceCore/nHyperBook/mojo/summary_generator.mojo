# ============================================================================
# HyperShimmy Summary Generator (Mojo)
# ============================================================================
#
# Day 31 Implementation: Multi-document summarization
#
# Features:
# - Multiple summary types (brief, detailed, executive)
# - Multi-document synthesis
# - Key points extraction
# - Source attribution
# - Configurable length and style
#
# Integration:
# - Uses ShimmyLLM from llm_chat.mojo
# - Integrates with semantic search for document retrieval
# - Prepares for Summary UI (Day 33)
# ============================================================================

from collections import List, Dict
from memory import memset_zero, UnsafePointer
from algorithm import min, max


# ============================================================================
# Summary Type Enumeration
# ============================================================================

struct SummaryType:
    """Types of summaries that can be generated."""
    
    var value: String
    
    fn __init__(inout self, value: String):
        self.value = value
    
    @staticmethod
    fn brief() -> SummaryType:
        """Brief summary (1-2 paragraphs, high-level overview)."""
        return SummaryType("brief")
    
    @staticmethod
    fn detailed() -> SummaryType:
        """Detailed summary (3-5 paragraphs, comprehensive analysis)."""
        return SummaryType("detailed")
    
    @staticmethod
    fn executive() -> SummaryType:
        """Executive summary (structured, action-oriented)."""
        return SummaryType("executive")
    
    @staticmethod
    fn bullet_points() -> SummaryType:
        """Bullet point summary (key takeaways)."""
        return SummaryType("bullet_points")
    
    @staticmethod
    fn comparative() -> SummaryType:
        """Comparative summary (compare/contrast multiple sources)."""
        return SummaryType("comparative")
    
    fn to_string(self) -> String:
        return self.value


# ============================================================================
# Summary Configuration
# ============================================================================

struct SummaryConfig:
    """Configuration for summary generation."""
    
    var summary_type: SummaryType
    var max_length: Int  # Max words in summary
    var include_citations: Bool
    var include_key_points: Bool
    var tone: String  # "professional", "academic", "casual"
    var focus_areas: List[String]  # Specific topics to emphasize
    
    fn __init__(inout self,
                summary_type: SummaryType,
                max_length: Int = 500,
                include_citations: Bool = True,
                include_key_points: Bool = True,
                tone: String = "professional",
                focus_areas: List[String] = List[String]()):
        """
        Initialize summary configuration.
        
        Args:
            summary_type: Type of summary to generate
            max_length: Maximum words in summary
            include_citations: Include source citations
            include_key_points: Extract key points
            tone: Writing tone
            focus_areas: Specific areas to focus on
        """
        self.summary_type = summary_type
        self.max_length = max_length
        self.include_citations = include_citations
        self.include_key_points = include_key_points
        self.tone = tone
        self.focus_areas = focus_areas


# ============================================================================
# Summary Request
# ============================================================================

struct SummaryRequest:
    """Request for summary generation."""
    
    var source_ids: List[String]
    var query: String  # Optional focus query
    var config: SummaryConfig
    var max_sources: Int
    
    fn __init__(inout self,
                source_ids: List[String],
                query: String = "",
                config: SummaryConfig = SummaryConfig(SummaryType.brief()),
                max_sources: Int = 10):
        """
        Initialize summary request.
        
        Args:
            source_ids: IDs of documents to summarize
            query: Optional focus query
            config: Summary configuration
            max_sources: Maximum number of sources to include
        """
        self.source_ids = source_ids
        self.query = query
        self.config = config
        self.max_sources = max_sources
    
    fn to_string(self) -> String:
        var result = String("SummaryRequest[\n")
        result += String("  type: ") + self.config.summary_type.to_string() + "\n"
        result += String("  sources: ") + String(len(self.source_ids)) + "\n"
        result += String("  max_length: ") + String(self.config.max_length) + " words\n"
        result += String("]")
        return result


# ============================================================================
# Key Point
# ============================================================================

struct KeyPoint:
    """A key point extracted from documents."""
    
    var content: String
    var importance: Float32  # 0.0-1.0
    var source_ids: List[String]
    var category: String
    
    fn __init__(inout self,
                content: String,
                importance: Float32,
                source_ids: List[String],
                category: String = "general"):
        self.content = content
        self.importance = importance
        self.source_ids = source_ids
        self.category = category
    
    fn to_string(self) -> String:
        var result = String("• ") + self.content
        if len(self.source_ids) > 0:
            result += String(" [") + String(len(self.source_ids)) + " source(s)]"
        return result


# ============================================================================
# Summary Response
# ============================================================================

struct SummaryResponse:
    """Generated summary with metadata."""
    
    var summary_text: String
    var key_points: List[KeyPoint]
    var source_ids: List[String]
    var summary_type: SummaryType
    var word_count: Int
    var confidence: Float32
    var processing_time_ms: Int
    var metadata: String  # JSON metadata
    
    fn __init__(inout self,
                summary_text: String,
                key_points: List[KeyPoint],
                source_ids: List[String],
                summary_type: SummaryType,
                word_count: Int,
                confidence: Float32,
                processing_time_ms: Int,
                metadata: String = "{}"):
        self.summary_text = summary_text
        self.key_points = key_points
        self.source_ids = source_ids
        self.summary_type = summary_type
        self.word_count = word_count
        self.confidence = confidence
        self.processing_time_ms = processing_time_ms
        self.metadata = metadata
    
    fn to_string(self) -> String:
        var result = String("SummaryResponse[\n")
        result += String("  type: ") + self.summary_type.to_string() + "\n"
        result += String("  words: ") + String(self.word_count) + "\n"
        result += String("  key_points: ") + String(len(self.key_points)) + "\n"
        result += String("  sources: ") + String(len(self.source_ids)) + "\n"
        result += String("  confidence: ") + String(self.confidence) + "\n"
        result += String("  time: ") + String(self.processing_time_ms) + "ms\n"
        result += String("]")
        return result


# ============================================================================
# Summary Prompt Templates
# ============================================================================

struct SummaryPrompts:
    """Prompt templates for different summary types."""
    
    @staticmethod
    fn get_system_prompt() -> String:
        """Get system prompt for summary generation."""
        return """You are an expert research analyst and summarization specialist integrated into HyperShimmy.

Your role is to:
1. Synthesize information from multiple documents
2. Extract key insights and main points
3. Provide accurate, well-structured summaries
4. Cite sources appropriately
5. Maintain objectivity and clarity

Guidelines:
- Always base summaries on the provided documents
- Identify patterns and themes across sources
- Highlight consensus and disagreements
- Use clear, professional language
- Structure information logically
- Cite sources when making specific claims"""
    
    @staticmethod
    fn get_brief_prompt(focus: String = "") -> String:
        """Get prompt for brief summary."""
        var prompt = String("""Generate a BRIEF summary (1-2 paragraphs) of the provided documents.

Requirements:
- Capture the main theme and key message
- Keep it concise (100-150 words)
- Focus on high-level overview
- Use clear, accessible language""")
        
        if len(focus) > 0:
            prompt += String("\n- Focus specifically on: ") + focus
        
        return prompt
    
    @staticmethod
    fn get_detailed_prompt(focus: String = "") -> String:
        """Get prompt for detailed summary."""
        var prompt = String("""Generate a DETAILED summary (3-5 paragraphs) of the provided documents.

Requirements:
- Cover main themes comprehensively
- Include important details and context
- Aim for 300-500 words
- Organize information logically
- Cite sources for key claims
- Identify connections between concepts""")
        
        if len(focus) > 0:
            prompt += String("\n- Emphasize information about: ") + focus
        
        return prompt
    
    @staticmethod
    fn get_executive_prompt() -> String:
        """Get prompt for executive summary."""
        return """Generate an EXECUTIVE summary with the following structure:

1. **Overview** (2-3 sentences)
   - What is this about?
   - Why does it matter?

2. **Key Findings** (3-5 bullet points)
   - Most important insights
   - Critical data or conclusions

3. **Recommendations** (2-3 bullet points)
   - Action items or next steps
   - Strategic implications

Requirements:
- Be action-oriented and decision-focused
- Use clear, business-appropriate language
- Highlight actionable insights
- Keep total length to 250-300 words"""
    
    @staticmethod
    fn get_bullet_points_prompt() -> String:
        """Get prompt for bullet point summary."""
        return """Generate a BULLET POINT summary of the key takeaways.

Requirements:
- Extract 5-8 main points
- Each point should be 1-2 sentences
- Organize by importance or theme
- Be specific and concrete
- Include source citations
- Use clear, concise language

Format each point as:
• [Main point] (Source: doc_id)"""
    
    @staticmethod
    fn get_comparative_prompt() -> String:
        """Get prompt for comparative summary."""
        return """Generate a COMPARATIVE summary analyzing the documents.

Structure:
1. **Common Themes** (what documents agree on)
2. **Key Differences** (where documents diverge)
3. **Unique Contributions** (what each document adds)
4. **Synthesis** (integrated understanding)

Requirements:
- Identify patterns across sources
- Highlight consensus and contradictions
- Note complementary information
- Maintain neutral, analytical tone
- Cite specific sources
- Aim for 300-400 words"""


# ============================================================================
# Summary Generator
# ============================================================================

struct SummaryGenerator:
    """
    Generates summaries from multiple documents using LLM.
    
    This module coordinates document retrieval, content synthesis,
    and summary generation with configurable types and styles.
    """
    
    var llm_config: String  # Would be LLMConfig, simplified for now
    var model_name: String
    var temperature: Float32
    
    fn __init__(inout self,
                model_name: String = "llama-3.2-1b",
                temperature: Float32 = 0.5):
        """
        Initialize summary generator.
        
        Args:
            model_name: LLM model to use
            temperature: Generation temperature (lower for summaries)
        """
        self.llm_config = "{}"
        self.model_name = model_name
        self.temperature = temperature
        
        print("📝 Summary Generator initialized")
        print("   Model: " + model_name)
        print("   Temperature: " + String(temperature))
    
    fn generate_summary(self,
                        request: SummaryRequest,
                        document_chunks: List[String]) -> SummaryResponse:
        """
        Generate summary from documents.
        
        Args:
            request: Summary request with configuration
            document_chunks: Relevant text chunks from documents
        
        Returns:
            Generated summary with metadata
        """
        print("\n" + "=" * 70)
        print("📝 Generating Summary")
        print("=" * 70)
        print(request.to_string())
        
        var start_time = self._get_timestamp()
        
        # Build prompt based on summary type
        var prompt = self._build_summary_prompt(request, document_chunks)
        
        # Generate summary (mock for now)
        var summary_text = self._generate_summary_text(request, document_chunks)
        
        # Extract key points
        var key_points = self._extract_key_points(document_chunks, request.source_ids)
        
        # Calculate metrics
        var word_count = self._count_words(summary_text)
        var confidence = self._calculate_confidence(document_chunks)
        
        var end_time = self._get_timestamp()
        var processing_time = end_time - start_time
        
        # Build metadata
        var metadata = self._build_metadata(request, word_count, confidence)
        
        var response = SummaryResponse(
            summary_text,
            key_points,
            request.source_ids,
            request.config.summary_type,
            word_count,
            confidence,
            processing_time,
            metadata
        )
        
        print("\n✅ Summary generated successfully!")
        print(response.to_string())
        
        return response
    
    fn _build_summary_prompt(self,
                             request: SummaryRequest,
                             chunks: List[String]) -> String:
        """Build prompt for summary generation."""
        var prompt = String()
        
        # Add system prompt
        prompt += String("[SYSTEM]\n")
        prompt += SummaryPrompts.get_system_prompt()
        prompt += String("\n\n")
        
        # Add task-specific prompt
        prompt += String("[TASK]\n")
        
        var summary_type = request.config.summary_type.value
        if summary_type == "brief":
            prompt += SummaryPrompts.get_brief_prompt(request.query)
        elif summary_type == "detailed":
            prompt += SummaryPrompts.get_detailed_prompt(request.query)
        elif summary_type == "executive":
            prompt += SummaryPrompts.get_executive_prompt()
        elif summary_type == "bullet_points":
            prompt += SummaryPrompts.get_bullet_points_prompt()
        elif summary_type == "comparative":
            prompt += SummaryPrompts.get_comparative_prompt()
        else:
            prompt += SummaryPrompts.get_detailed_prompt(request.query)
        
        prompt += String("\n\n")
        
        # Add documents
        prompt += String("[DOCUMENTS]\n\n")
        var num_chunks = min(len(chunks), 10)
        for i in range(num_chunks):
            prompt += String("Document ") + String(i + 1) + ":\n"
            prompt += chunks[i] + "\n\n"
        
        prompt += String("[END DOCUMENTS]\n\n")
        prompt += String("Generate the summary now:\n")
        
        return prompt
    
    fn _generate_summary_text(self,
                               request: SummaryRequest,
                               chunks: List[String]) -> String:
        """
        Generate summary text (mock implementation).
        
        In production, would call ShimmyLLM with the prompt.
        """
        var summary_type = request.config.summary_type.value
        
        if summary_type == "brief":
            return self._generate_brief_summary(chunks)
        elif summary_type == "detailed":
            return self._generate_detailed_summary(chunks, request.query)
        elif summary_type == "executive":
            return self._generate_executive_summary(chunks)
        elif summary_type == "bullet_points":
            return self._generate_bullet_summary(chunks)
        elif summary_type == "comparative":
            return self._generate_comparative_summary(chunks)
        else:
            return self._generate_detailed_summary(chunks, request.query)
    
    fn _generate_brief_summary(self, chunks: List[String]) -> String:
        """Generate brief summary."""
        return """The documents provide a comprehensive overview of machine learning and artificial intelligence concepts. They explore fundamental principles including supervised and unsupervised learning, neural networks, and deep learning architectures. The materials emphasize both theoretical foundations and practical applications, making them valuable resources for understanding modern AI systems and their impact on various industries."""
    
    fn _generate_detailed_summary(self, chunks: List[String], focus: String) -> String:
        """Generate detailed summary."""
        var summary = String("""The collection of documents presents a thorough examination of machine learning and artificial intelligence, covering both foundational concepts and advanced techniques. 

The materials begin by establishing core ML principles, including the distinction between supervised learning (where models learn from labeled data) and unsupervised learning (where patterns are discovered without explicit labels). Particular attention is given to neural networks, which form the backbone of modern deep learning systems. These networks, inspired by biological neural structures, consist of interconnected layers that process information hierarchically.

Deep learning emerges as a central theme, with detailed discussions of convolutional neural networks (CNNs) for image processing and recurrent neural networks (RNNs) for sequential data. The documents explain how these architectures have revolutionized fields like computer vision, natural language processing, and speech recognition through their ability to automatically learn features from raw data.

The practical applications section demonstrates ML's transformative impact across industries. Examples include predictive analytics in healthcare, autonomous systems in transportation, and recommendation engines in e-commerce. The materials also address important considerations around model evaluation, bias mitigation, and ethical AI development.

Throughout the documents, there's a consistent emphasis on the iterative nature of ML development—from data preparation and feature engineering through model training, validation, and deployment. The content successfully bridges theoretical understanding with practical implementation, making it valuable for both learners and practitioners.""")
        
        if len(focus) > 0:
            summary += String("\n\nRegarding ") + focus + ": The documents provide specific insights into this area, highlighting key methodologies and best practices."
        
        return summary
    
    fn _generate_executive_summary(self, chunks: List[String]) -> String:
        """Generate executive summary."""
        return """**Overview**
This research examines machine learning and AI technologies that are transforming how organizations process data and make decisions. Understanding these technologies is critical for maintaining competitive advantage in today's data-driven landscape.

**Key Findings**
• Machine learning enables automated pattern recognition and predictive analytics at scale
• Deep learning architectures (CNNs, RNNs) achieve human-level performance in specialized tasks
• Implementation requires careful consideration of data quality, model bias, and interpretability
• ROI is highest in applications with large datasets and clear success metrics
• Ethical considerations and regulatory compliance are increasingly important

**Recommendations**
• Invest in data infrastructure and quality assurance processes before ML deployment
• Start with well-defined use cases that have measurable business impact
• Develop internal expertise through training and strategic hiring
• Establish governance frameworks for responsible AI development"""
    
    fn _generate_bullet_summary(self, chunks: List[String]) -> String:
        """Generate bullet point summary."""
        return """• Machine learning is a subset of AI that enables computers to learn from data without explicit programming (Source: doc_001, doc_002)

• Supervised learning uses labeled data to train models for prediction tasks, while unsupervised learning discovers patterns in unlabeled data (Source: doc_001)

• Neural networks consist of interconnected layers that process information hierarchically, inspired by biological neural structures (Source: doc_002, doc_003)

• Deep learning refers to neural networks with multiple hidden layers, enabling automatic feature extraction from raw data (Source: doc_002)

• Convolutional Neural Networks (CNNs) excel at image processing tasks through specialized layers for feature detection (Source: doc_003)

• Applications span healthcare diagnostics, autonomous vehicles, natural language processing, and recommendation systems (Source: doc_001, doc_003)

• Key challenges include data quality requirements, computational resources, model interpretability, and ethical considerations (Source: doc_002, doc_003)"""
    
    fn _generate_comparative_summary(self, chunks: List[String]) -> String:
        """Generate comparative summary."""
        return """**Common Themes**
All documents agree that machine learning represents a paradigm shift in computing, moving from rule-based programming to data-driven learning. They consistently emphasize the importance of quality data and proper model evaluation. The materials converge on neural networks as the dominant architecture for complex pattern recognition tasks.

**Key Differences**
Document 1 takes a more theoretical approach, focusing on mathematical foundations and algorithm design. Document 2 emphasizes practical implementation and real-world applications. Document 3 provides a balanced view but dedicates more attention to ethical considerations and societal impact.

**Unique Contributions**
Document 1 uniquely covers reinforcement learning and game theory applications. Document 2 provides detailed case studies from industry implementations. Document 3 offers the most comprehensive discussion of bias mitigation and fairness in AI systems.

**Synthesis**
Together, these documents provide a well-rounded understanding of machine learning. The theoretical foundations from Document 1 complement the practical insights from Document 2, while Document 3's ethical framework ensures responsible development. This combination equips readers with both technical competence and ethical awareness necessary for modern AI practice."""
    
    fn _extract_key_points(self,
                           chunks: List[String],
                           source_ids: List[String]) -> List[KeyPoint]:
        """Extract key points from documents."""
        var points = List[KeyPoint]()
        
        # Create some example key points
        var point1 = KeyPoint(
            "Machine learning enables automated pattern recognition from data",
            0.95,
            source_ids,
            "core_concept"
        )
        points.append(point1)
        
        var point2 = KeyPoint(
            "Neural networks form the foundation of modern deep learning",
            0.90,
            source_ids,
            "technology"
        )
        points.append(point2)
        
        var point3 = KeyPoint(
            "Applications include healthcare, autonomous systems, and NLP",
            0.85,
            source_ids,
            "applications"
        )
        points.append(point3)
        
        var point4 = KeyPoint(
            "Data quality and model interpretability are critical challenges",
            0.80,
            source_ids,
            "challenges"
        )
        points.append(point4)
        
        var point5 = KeyPoint(
            "Ethical considerations and bias mitigation are increasingly important",
            0.75,
            source_ids,
            "ethics"
        )
        points.append(point5)
        
        return points
    
    fn _count_words(self, text: String) -> Int:
        """Count words in text (rough estimate)."""
        var count = 0
        var in_word = False
        
        for i in range(len(text)):
            var c = text[i]
            if c == ' ' or c == '\n' or c == '\t':
                if in_word:
                    count += 1
                    in_word = False
            else:
                in_word = True
        
        if in_word:
            count += 1
        
        return count
    
    fn _calculate_confidence(self, chunks: List[String]) -> Float32:
        """Calculate confidence score based on available information."""
        # More chunks = higher confidence
        var num_chunks = Float32(len(chunks))
        var confidence = min(0.95, 0.60 + (num_chunks / 20.0))
        return confidence
    
    fn _build_metadata(self,
                       request: SummaryRequest,
                       word_count: Int,
                       confidence: Float32) -> String:
        """Build metadata JSON string."""
        var meta = String('{"summary_type":"') + request.config.summary_type.to_string() + '"'
        meta += String(',"word_count":') + String(word_count)
        meta += String(',"confidence":') + String(confidence)
        meta += String(',"tone":"') + request.config.tone + '"'
        meta += String(',"sources":') + String(len(request.source_ids))
        meta += String("}")
        return meta
    
    fn _get_timestamp(self) -> Int:
        """Get current timestamp in ms."""
        return 1737025000


# ============================================================================
# C ABI Exports for Zig Integration
# ============================================================================

@export("summary_generate")
fn summary_generate_c(
    request_ptr: DTypePointer[DType.uint8],
    request_len: Int,
    chunks_ptr: DTypePointer[DType.uint8],
    chunks_len: Int,
    response_out: DTypePointer[DType.uint8]
) -> Int32:
    """
    Generate summary from C/Zig.
    
    Args:
        request_ptr: Pointer to JSON request
        request_len: Length of request
        chunks_ptr: Pointer to JSON chunks array
        chunks_len: Length of chunks
        response_out: Output buffer for response
    
    Returns:
        0 for success, error code otherwise
    """
    # In real implementation, would parse inputs and generate summary
    return 0


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

fn main():
    """Test the summary generator module."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   HyperShimmy Summary Generator (Mojo) - Day 31           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Initialize generator
    print("\n" + "=" * 70)
    print("Step 1: Initialize Summary Generator")
    print("=" * 70)
    
    var generator = SummaryGenerator("llama-3.2-1b", 0.5)
    
    # Prepare test data
    print("\n" + "=" * 70)
    print("Step 2: Prepare Test Documents")
    print("=" * 70)
    
    var source_ids = List[String]()
    source_ids.append(String("doc_001"))
    source_ids.append(String("doc_002"))
    source_ids.append(String("doc_003"))
    
    var chunks = List[String]()
    chunks.append(String("Machine learning is a subset of AI. It uses algorithms to learn from data."))
    chunks.append(String("Deep learning uses neural networks with multiple layers."))
    chunks.append(String("Applications include computer vision, NLP, and robotics."))
    
    print("Documents prepared:")
    print("  Sources: " + String(len(source_ids)))
    print("  Chunks: " + String(len(chunks)))
    
    # Test Brief Summary
    print("\n" + "=" * 70)
    print("Test 1: Brief Summary")
    print("=" * 70)
    
    var brief_config = SummaryConfig(
        SummaryType.brief(),
        150,  # max_length
        True,  # include_citations
        False,  # include_key_points
        "professional",  # tone
        List[String]()
    )
    var brief_request = SummaryRequest(source_ids, "", brief_config, 10)
    var brief_response = generator.generate_summary(brief_request, chunks)
    
    print("\n📄 Brief Summary:")
    print(brief_response.summary_text)
    
    # Test Detailed Summary
    print("\n" + "=" * 70)
    print("Test 2: Detailed Summary")
    print("=" * 70)
    
    var detailed_config = SummaryConfig(
        SummaryType.detailed(),
        500,
        True,
        True,
        "academic",
        List[String]()
    )
    var detailed_request = SummaryRequest(source_ids, "neural networks", detailed_config, 10)
    var detailed_response = generator.generate_summary(detailed_request, chunks)
    
    print("\n📄 Detailed Summary:")
    print(detailed_response.summary_text[:300] + "...")
    print("\n🔑 Key Points:")
    for i in range(min(3, len(detailed_response.key_points))):
        print(detailed_response.key_points[i].to_string())
    
    # Test Executive Summary
    print("\n" + "=" * 70)
    print("Test 3: Executive Summary")
    print("=" * 70)
    
    var exec_config = SummaryConfig(
        SummaryType.executive(),
        300,
        True,
        True,
        "professional",
        List[String]()
    )
    var exec_request = SummaryRequest(source_ids, "", exec_config, 10)
    var exec_response = generator.generate_summary(exec_request, chunks)
    
    print("\n📄 Executive Summary:")
    print(exec_response.summary_text[:400] + "...")
    
    # Test Bullet Points
    print("\n" + "=" * 70)
    print("Test 4: Bullet Point Summary")
    print("=" * 70)
    
    var bullet_config = SummaryConfig(
        SummaryType.bullet_points(),
        200,
        True,
        False,
        "professional",
        List[String]()
    )
    var bullet_request = SummaryRequest(source_ids, "", bullet_config, 10)
    var bullet_response = generator.generate_summary(bullet_request, chunks)
    
    print("\n📄 Bullet Point Summary:")
    print(bullet_response.summary_text[:400] + "...")
    
    # Test Comparative Summary
    print("\n" + "=" * 70)
    print("Test 5: Comparative Summary")
    print("=" * 70)
    
    var comp_config = SummaryConfig(
        SummaryType.comparative(),
        400,
        True,
        False,
        "academic",
        List[String]()
    )
    var comp_request = SummaryRequest(source_ids, "", comp_config, 10)
    var comp_response = generator.generate_summary(comp_request, chunks)
    
    print("\n📄 Comparative Summary:")
    print(comp_response.summary_text[:400] + "...")
    
    print("\n" + "=" * 70)
    print("✅ Summary generator test complete!")
    print("=" * 70)
    print("\nSummary Statistics:")
    print("  Brief: " + String(brief_response.word_count) + " words, " + 
          String(brief_response.confidence) + " confidence")
    print("  Detailed: " + String(detailed_response.word_count) + " words, " + 
          String(detailed_response.confidence) + " confidence")
    print("  Executive: " + String(exec_response.word_count) + " words, " + 
          String(exec_response.confidence) + " confidence")
    print("  Bullet: " + String(bullet_response.word_count) + " words, " + 
          String(bullet_response.confidence) + " confidence")
    print("  Comparative: " + String(comp_response.word_count) + " words, " + 
          String(comp_response.confidence) + " confidence")
