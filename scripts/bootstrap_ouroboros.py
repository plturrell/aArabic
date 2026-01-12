#!/usr/bin/env python3
"""
OUROBOROS KNOWLEDGE INFRASTRUCTURE BOOTSTRAP
═════════════════════════════════════════════════════════════════

Initializes and verifies the complete Ouroboros self-improving system:
- Marqo (semantic vector search)
- Qdrant (high-performance vectors)
- Memgraph (graph relationships)
- Kafka (real-time event streaming)

Usage:
    python scripts/bootstrap_ouroboros.py [--verify-only]
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from serviceData.knowledge_bridge import (
    get_knowledge_bridge,
    run_ouroboros_cycle,
    OuroborosEventProcessor
)
from serviceData.serviceMarqo.adapter import get_marqo_adapter
from serviceData.serviceKafka.adapter import get_kafka_adapter

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OuroborosBootstrap:
    """Bootstrap and verify Ouroboros knowledge infrastructure"""
    
    def __init__(self):
        self.bridge = None
        self.health_status = {}
    
    async def initialize(self):
        """Initialize all components"""
        print("\n" + "="*70)
        print("🌀 OUROBOROS KNOWLEDGE INFRASTRUCTURE BOOTSTRAP")
        print("="*70 + "\n")
        
        logger.info("Step 1: Creating Knowledge Bridge...")
        self.bridge = get_knowledge_bridge()
        
        logger.info("Step 2: Checking health of all systems...")
        self.health_status = await self.bridge.health_check()
        
        # Display health status
        print("\n📊 SYSTEM HEALTH STATUS:")
        print("-" * 70)
        for system, status in self.health_status.items():
            status_icon = "✅" if status.get("status") in ["healthy", "connected"] else "❌"
            print(f"{status_icon} {system.upper():15} {status.get('status', 'unknown')}")
        print("-" * 70)
        
        # Check if all systems are healthy
        all_healthy = all(
            s.get("status") in ["healthy", "connected"] 
            for s in self.health_status.values()
        )
        
        if not all_healthy:
            logger.error("⚠️  Some systems are not healthy. Please check the services.")
            logger.info("Hint: Run 'docker-compose -f docker/compose/docker-compose.knowledge.yml up -d'")
            return False
        
        logger.info("Step 3: Initializing infrastructure...")
        result = await self.bridge.initialize_infrastructure()
        
        if result.get("status") == "initialized":
            logger.info("✅ All knowledge infrastructure initialized successfully!")
            return True
        else:
            logger.error("❌ Failed to initialize infrastructure")
            return False
    
    async def verify_integration(self):
        """Verify that all systems are properly integrated"""
        print("\n" + "="*70)
        print("🔍 VERIFYING OUROBOROS INTEGRATION")
        print("="*70 + "\n")
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Marqo indexing
        print("Test 1: Marqo semantic search...")
        tests_total += 1
        try:
            await self.bridge.marqo.add_documents(
                "semantic_knowledge",
                [{
                    "_id": "test_doc_001",
                    "content": "Ouroboros self-improving AI system",
                    "description": "A system that eats its own tail to improve"
                }]
            )
            results = await self.bridge.marqo.search(
                "semantic_knowledge",
                "self-improving artificial intelligence",
                limit=1
            )
            if results.get("hits"):
                print("  ✅ Marqo semantic search working")
                tests_passed += 1
            else:
                print("  ⚠️  Marqo search returned no results")
        except Exception as e:
            print(f"  ❌ Marqo test failed: {e}")
        
        # Test 2: Kafka event streaming
        print("\nTest 2: Kafka event streaming...")
        tests_total += 1
        try:
            self.bridge.kafka.log_model_prediction(
                model_name="test_model",
                input_data={"test": "input"},
                prediction="test output",
                confidence=0.95
            )
            self.bridge.kafka.flush()
            print("  ✅ Kafka event streaming working")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Kafka test failed: {e}")
        
        # Test 3: Cross-system search
        print("\nTest 3: Unified cross-system search...")
        tests_total += 1
        try:
            results = await self.bridge.unified_search(
                "self-improving system",
                search_type="hybrid"
            )
            if any(results.values()):
                print("  ✅ Cross-system search working")
                tests_passed += 1
            else:
                print("  ⚠️  Cross-system search returned no results")
        except Exception as e:
            print(f"  ❌ Cross-system search failed: {e}")
        
        # Test 4: Ouroboros cycle (dry run)
        print("\nTest 4: Ouroboros improvement cycle...")
        tests_total += 1
        try:
            task = {
                "id": "bootstrap_test",
                "function": "verification",
                "inputs": {"test": "bootstrap"}
            }
            
            cycle_results = await run_ouroboros_cycle(
                task=task,
                model_name="bootstrap_model",
                iterations=1
            )
            
            if cycle_results.get("cycles"):
                print("  ✅ Ouroboros cycle executed successfully")
                tests_passed += 1
            else:
                print("  ⚠️  Ouroboros cycle incomplete")
        except Exception as e:
            print(f"  ❌ Ouroboros cycle failed: {e}")
        
        # Summary
        print("\n" + "="*70)
        print(f"VERIFICATION SUMMARY: {tests_passed}/{tests_total} tests passed")
        print("="*70 + "\n")
        
        if tests_passed == tests_total:
            print("🎉 ALL SYSTEMS OPERATIONAL - OUROBOROS READY!")
            return True
        elif tests_passed > 0:
            print("⚠️  PARTIAL INTEGRATION - Some systems need attention")
            return False
        else:
            print("❌ INTEGRATION FAILED - Please check system logs")
            return False
    
    async def demonstrate_cycle(self):
        """Demonstrate a complete Ouroboros improvement cycle"""
        print("\n" + "="*70)
        print("🌀 DEMONSTRATING OUROBOROS SELF-IMPROVEMENT CYCLE")
        print("="*70 + "\n")
        
        task = {
            "id": "demo_task_001",
            "function": "code_generation",
            "inputs": {
                "prompt": "Generate a Python function to classify Arabic text sentiment"
            }
        }
        
        logger.info("Running 3 improvement iterations...")
        results = await run_ouroboros_cycle(
            task=task,
            model_name="qwen3-coder",
            iterations=3
        )
        
        print("\n" + "="*70)
        print("📊 CYCLE RESULTS")
        print("="*70)
        print(f"Total iterations: {results['total_iterations']}")
        print(f"Cumulative improvement: {results['final_improvement']:.2%}")
        print("\nPer-iteration breakdown:")
        for cycle in results['cycles']:
            iteration = cycle['iteration']
            improvement = cycle['learning'].get('metric', 0.0)
            print(f"  Iteration {iteration}: {improvement:.2%} improvement")
        print("="*70 + "\n")
    
    async def cleanup(self):
        """Clean up connections"""
        if self.bridge:
            await self.bridge.close()
            logger.info("🧹 Cleaned up all connections")


async def main():
    """Main bootstrap process"""
    parser = argparse.ArgumentParser(
        description="Bootstrap Ouroboros Knowledge Infrastructure"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing installation without re-initializing"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration of Ouroboros cycle"
    )
    
    args = parser.parse_args()
    
    bootstrap = OuroborosBootstrap()
    
    try:
        # Initialize (unless verify-only)
        if not args.verify_only:
            success = await bootstrap.initialize()
            if not success:
                print("\n❌ Bootstrap failed. Please check the logs and try again.")
                return 1
        
        # Verify integration
        success = await bootstrap.verify_integration()
        
        if not success:
            print("\n⚠️  Verification incomplete. Some systems may need attention.")
            return 1
        
        # Run demonstration if requested
        if args.demo:
            await bootstrap.demonstrate_cycle()
        
        print("\n✅ OUROBOROS BOOTSTRAP COMPLETE!")
        print("\nNext steps:")
        print("  1. Start using the Knowledge Bridge in your code")
        print("  2. Stream events to Kafka for real-time learning")
        print("  3. Query across all knowledge systems")
        print("  4. Run Ouroboros cycles for continuous improvement")
        print("\nExample:")
        print("  from serviceData.knowledge_bridge import run_ouroboros_cycle")
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Bootstrap interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Bootstrap failed with error: {e}", exc_info=True)
        return 1
    finally:
        await bootstrap.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
