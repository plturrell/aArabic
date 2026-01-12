"""
Kafka Adapter - Event Streaming for Real-time Learning
Enables the Ouroboros system to stream and process learning events in real-time
"""

from typing import Dict, List, Optional, Any, Callable
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError
import json
import asyncio
from datetime import datetime


class KafkaAdapter:
    """
    Adapter for Apache Kafka event streaming
    Handles real-time learning events for the Ouroboros cycle
    """
    
    # Ouroboros Event Topics
    TOPICS = {
        "MODEL_PREDICTIONS": "model_predictions",
        "LEARNING_EVENTS": "learning_events",
        "PROOF_VERIFICATION": "proof_verification",
        "CODE_CHANGES": "code_changes",
        "EXECUTION_TRACES": "execution_traces",
        "FEEDBACK_LOOPS": "feedback_loops",
        "SEMANTIC_UPDATES": "semantic_updates",
        "GRAPH_CHANGES": "graph_changes"
    }
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: str = "nucleus-ouroboros"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        
        # Initialize producer with JSON serialization
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=f"{client_id}-producer",
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3
        )
        
        # Admin client for topic management
        self.admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id=f"{client_id}-admin"
        )
        
        self.consumers: Dict[str, KafkaConsumer] = {}
    
    def create_topics(self, num_partitions: int = 1, replication_factor: int = 1):
        """Create all Ouroboros topics"""
        topics = [
            NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )
            for topic_name in self.TOPICS.values()
        ]
        
        try:
            result = self.admin.create_topics(topics, validate_only=False)
            return {"status": "created", "topics": list(self.TOPICS.values())}
        except TopicAlreadyExistsError:
            return {"status": "exists", "topics": list(self.TOPICS.values())}
    
    def send_event(
        self,
        topic: str,
        event: Dict[str, Any],
        key: Optional[str] = None
    ) -> None:
        """
        Send an event to a Kafka topic
        
        Args:
            topic: Topic name
            event: Event payload
            key: Optional partition key
        """
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.utcnow().isoformat()
        
        future = self.producer.send(topic, value=event, key=key)
        
        # Non-blocking send
        future.add_callback(self._on_send_success)
        future.add_errback(self._on_send_error)
    
    def _on_send_success(self, record_metadata):
        """Callback for successful send"""
        print(f"✓ Event sent to {record_metadata.topic}:{record_metadata.partition}")
    
    def _on_send_error(self, exc):
        """Callback for send error"""
        print(f"✗ Error sending event: {exc}")
    
    def flush(self):
        """Flush all pending messages"""
        self.producer.flush()
    
    def create_consumer(
        self,
        topics: List[str],
        group_id: str,
        auto_offset_reset: str = 'earliest'
    ) -> KafkaConsumer:
        """
        Create a consumer for specified topics
        
        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            auto_offset_reset: Where to start reading
        """
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True,
            auto_commit_interval_ms=5000
        )
        
        self.consumers[group_id] = consumer
        return consumer
    
    def consume_events(
        self,
        consumer: KafkaConsumer,
        callback: Callable[[Dict[str, Any]], None],
        max_messages: Optional[int] = None
    ):
        """
        Consume events from topics
        
        Args:
            consumer: KafkaConsumer instance
            callback: Function to process each message
            max_messages: Max messages to consume (None = infinite)
        """
        count = 0
        try:
            for message in consumer:
                event = message.value
                event["_kafka_metadata"] = {
                    "topic": message.topic,
                    "partition": message.partition,
                    "offset": message.offset,
                    "timestamp": message.timestamp
                }
                
                callback(event)
                count += 1
                
                if max_messages and count >= max_messages:
                    break
        except KeyboardInterrupt:
            print("Stopping consumer...")
        finally:
            consumer.close()
    
    def close(self):
        """Close all connections"""
        self.producer.close()
        for consumer in self.consumers.values():
            consumer.close()
        self.admin.close()
    
    # ─────────────────────────────────────────────────────────────
    # Ouroboros-Specific Event Methods
    # ─────────────────────────────────────────────────────────────
    
    def log_model_prediction(
        self,
        model_name: str,
        input_data: Any,
        prediction: Any,
        confidence: float,
        metadata: Optional[Dict] = None
    ):
        """Log a model prediction event"""
        event = {
            "event_type": "model_prediction",
            "model_name": model_name,
            "input": str(input_data),
            "prediction": str(prediction),
            "confidence": confidence,
            "metadata": metadata or {}
        }
        self.send_event(self.TOPICS["MODEL_PREDICTIONS"], event, key=model_name)
    
    def log_learning_event(
        self,
        event_type: str,
        model_name: str,
        improvement_metric: float,
        details: Dict[str, Any]
    ):
        """Log a learning/improvement event"""
        event = {
            "event_type": event_type,
            "model_name": model_name,
            "improvement_metric": improvement_metric,
            "details": details
        }
        self.send_event(self.TOPICS["LEARNING_EVENTS"], event, key=model_name)
    
    def log_proof_verification(
        self,
        proof_id: str,
        theorem: str,
        verification_result: bool,
        proof_system: str,
        details: Dict[str, Any]
    ):
        """Log a proof verification event"""
        event = {
            "event_type": "proof_verification",
            "proof_id": proof_id,
            "theorem": theorem,
            "verified": verification_result,
            "proof_system": proof_system,
            "details": details
        }
        self.send_event(self.TOPICS["PROOF_VERIFICATION"], event, key=proof_id)
    
    def log_code_change(
        self,
        file_path: str,
        change_type: str,
        before: Optional[str],
        after: str,
        reason: str
    ):
        """Log a code change event"""
        event = {
            "event_type": "code_change",
            "file_path": file_path,
            "change_type": change_type,
            "before": before,
            "after": after,
            "reason": reason
        }
        self.send_event(self.TOPICS["CODE_CHANGES"], event, key=file_path)
    
    def log_execution_trace(
        self,
        execution_id: str,
        function_name: str,
        inputs: Dict[str, Any],
        outputs: Any,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Log an execution trace"""
        event = {
            "event_type": "execution_trace",
            "execution_id": execution_id,
            "function_name": function_name,
            "inputs": inputs,
            "outputs": str(outputs),
            "duration_ms": duration_ms,
            "success": success,
            "error": error
        }
        self.send_event(self.TOPICS["EXECUTION_TRACES"], event, key=execution_id)
    
    def log_feedback_loop(
        self,
        loop_id: str,
        iteration: int,
        metric_name: str,
        metric_value: float,
        improvement: float,
        metadata: Dict[str, Any]
    ):
        """Log feedback loop progress"""
        event = {
            "event_type": "feedback_loop",
            "loop_id": loop_id,
            "iteration": iteration,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "improvement": improvement,
            "metadata": metadata
        }
        self.send_event(self.TOPICS["FEEDBACK_LOOPS"], event, key=loop_id)
    
    def log_semantic_update(
        self,
        entity_id: str,
        entity_type: str,
        old_embedding: Optional[List[float]],
        new_embedding: List[float],
        reason: str
    ):
        """Log semantic embedding update"""
        event = {
            "event_type": "semantic_update",
            "entity_id": entity_id,
            "entity_type": entity_type,
            "embedding_changed": old_embedding is not None,
            "reason": reason
        }
        # Don't send full embeddings, just metadata
        self.send_event(self.TOPICS["SEMANTIC_UPDATES"], event, key=entity_id)
    
    def log_graph_change(
        self,
        change_type: str,
        node_id: Optional[str],
        edge_id: Optional[str],
        properties: Dict[str, Any]
    ):
        """Log graph database change"""
        event = {
            "event_type": "graph_change",
            "change_type": change_type,
            "node_id": node_id,
            "edge_id": edge_id,
            "properties": properties
        }
        key = node_id or edge_id or "unknown"
        self.send_event(self.TOPICS["GRAPH_CHANGES"], event, key=key)


# Singleton instance
_kafka_adapter: Optional[KafkaAdapter] = None


def get_kafka_adapter(
    bootstrap_servers: str = "localhost:9092",
    client_id: str = "nucleus-ouroboros"
) -> KafkaAdapter:
    """Get or create Kafka adapter instance"""
    global _kafka_adapter
    if _kafka_adapter is None:
        _kafka_adapter = KafkaAdapter(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id
        )
    return _kafka_adapter
