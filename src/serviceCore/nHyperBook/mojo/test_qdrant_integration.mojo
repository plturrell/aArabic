from embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingVector, BatchEmbeddingResult
from qdrant_bridge import QdrantConfig, EmbeddingPipeline
from collections import List

fn main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Qdrant Integration Test - Day 22                       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Step 1: Generate embeddings (Day 21)
    print("\n" + "=" * 60)
    print("Step 1: Generate Embeddings")
    print("=" * 60)
    
    var emb_config = EmbeddingConfig()
    var generator = EmbeddingGenerator(emb_config)
    _ = generator.load_model()
    
    var texts = List[String]()
    texts.append(String("Machine learning is a subset of AI"))
    texts.append(String("Neural networks are inspired by the brain"))
    texts.append(String("Deep learning uses multiple layers"))
    
    var chunk_ids = List[String]()
    chunk_ids.append(String("chunk_001"))
    chunk_ids.append(String("chunk_002"))
    chunk_ids.append(String("chunk_003"))
    
    var file_ids = List[String]()
    file_ids.append(String("file_1"))
    file_ids.append(String("file_1"))
    file_ids.append(String("file_1"))
    
    var indices = List[Int]()
    indices.append(0)
    indices.append(1)
    indices.append(2)
    
    var batch_result = generator.generate_batch(texts, chunk_ids, file_ids, indices)
    
    # Step 2: Setup Qdrant pipeline
    print("\n" + "=" * 60)
    print("Step 2: Setup Qdrant Pipeline")
    print("=" * 60)
    
    var qdrant_config = QdrantConfig()
    var pipeline = EmbeddingPipeline(qdrant_config)
    _ = pipeline.setup()
    
    # Step 3: Index embeddings
    print("\n" + "=" * 60)
    print("Step 3: Index Embeddings to Qdrant")
    print("=" * 60)
    
    var indexing_result = pipeline.process_and_index(batch_result)
    
    # Step 4: Print statistics
    print("\n" + "=" * 60)
    print("Step 4: Final Statistics")
    print("=" * 60)
    
    print(pipeline.get_stats())
    
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║              Integration Test Complete!                    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("")
    print("✅ Embeddings generated: " + String(len(batch_result.embeddings)))
    print("✅ Embeddings indexed: " + String(indexing_result.num_indexed))
    print("✅ Success rate: " + String(indexing_result.success_rate() * 100) + "%")
    print("")
