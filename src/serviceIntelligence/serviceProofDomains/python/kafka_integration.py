"""
Kafka Integration for Proof Domains

Provides:
1. Proof event streaming to Kafka
2. Proof event consumption from Kafka
3. Cross-domain proof linking via Kafka topics
4. Integration with n8n workflows

Requires: kafka-python or confluent-kafka
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from abc import ABC, abstractmethod

from proof_domains import (
    UnifiedProof, ProofEvent, ProofEventType,
    TOONEncoder, create_proof_event,
    MathProof, CodeProof, LanguageProof, DataProof,
    ProcessProof, InsightProof, RegulationProof,
    DOMAIN_MAP
)

logger = logging.getLogger(__name__)


# =============================================================================
# Kafka Configuration
# =============================================================================

@dataclass
class KafkaConfig:
    """Kafka connection configuration."""
    bootstrap_servers: str = "localhost:9092"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    client_id: str = "proof-domains"
    group_id: str = "proof-domains-consumer"

    # Topic configuration
    topic_prefix: str = "proofs"
    num_partitions: int = 7  # One per domain
    replication_factor: int = 1

    @property
    def topics(self) -> Dict[str, str]:
        """Get topic names for each domain."""
        return {
            "mathematics": f"{self.topic_prefix}.mathematics",
            "code": f"{self.topic_prefix}.code",
            "language": f"{self.topic_prefix}.language",
            "data": f"{self.topic_prefix}.data",
            "processes": f"{self.topic_prefix}.processes",
            "insights": f"{self.topic_prefix}.insights",
            "regulations": f"{self.topic_prefix}.regulations",
            "all": f"{self.topic_prefix}.all",  # Combined topic
            "links": f"{self.topic_prefix}.links",  # Cross-domain links
            "events": f"{self.topic_prefix}.events"  # All events
        }


# =============================================================================
# Abstract Kafka Client
# =============================================================================

class KafkaClient(ABC):
    """Abstract base class for Kafka operations."""

    @abstractmethod
    def produce(self, topic: str, key: str, value: Dict, headers: Dict = None) -> None:
        """Produce a message to Kafka."""
        pass

    @abstractmethod
    def consume(self, topics: List[str], handler: Callable[[Dict], None]) -> None:
        """Consume messages from Kafka topics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the Kafka connection."""
        pass


# =============================================================================
# Mock Kafka Client (for testing without Kafka)
# =============================================================================

class MockKafkaClient(KafkaClient):
    """Mock Kafka client for testing."""

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.messages: Dict[str, List[Dict]] = {topic: [] for topic in config.topics.values()}
        self.handlers: Dict[str, List[Callable]] = {}
        logger.info("Initialized MockKafkaClient")

    def produce(self, topic: str, key: str, value: Dict, headers: Dict = None) -> None:
        """Store message in memory."""
        message = {
            'topic': topic,
            'key': key,
            'value': value,
            'headers': headers or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        if topic not in self.messages:
            self.messages[topic] = []
        self.messages[topic].append(message)
        logger.info(f"Produced message to {topic}: {key}")

        # Trigger handlers
        if topic in self.handlers:
            for handler in self.handlers[topic]:
                handler(message)

    def consume(self, topics: List[str], handler: Callable[[Dict], None]) -> None:
        """Register handler for topics."""
        for topic in topics:
            if topic not in self.handlers:
                self.handlers[topic] = []
            self.handlers[topic].append(handler)
        logger.info(f"Registered consumer for topics: {topics}")

    def get_messages(self, topic: str) -> List[Dict]:
        """Get all messages for a topic."""
        return self.messages.get(topic, [])

    def close(self) -> None:
        """Clear handlers."""
        self.handlers.clear()
        logger.info("Closed MockKafkaClient")


# =============================================================================
# Real Kafka Client (requires kafka-python)
# =============================================================================

class RealKafkaClient(KafkaClient):
    """Real Kafka client using kafka-python."""

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = None
        self.consumer = None
        self._initialize()

    def _initialize(self):
        """Initialize Kafka producer and consumer."""
        try:
            from kafka import KafkaProducer, KafkaConsumer

            # Producer configuration
            producer_config = {
                'bootstrap_servers': self.config.bootstrap_servers,
                'client_id': f"{self.config.client_id}-producer",
                'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
                'key_serializer': lambda k: k.encode('utf-8') if k else None
            }

            if self.config.sasl_mechanism:
                producer_config.update({
                    'security_protocol': self.config.security_protocol,
                    'sasl_mechanism': self.config.sasl_mechanism,
                    'sasl_plain_username': self.config.sasl_username,
                    'sasl_plain_password': self.config.sasl_password
                })

            self.producer = KafkaProducer(**producer_config)
            logger.info(f"Connected to Kafka at {self.config.bootstrap_servers}")

        except ImportError:
            logger.warning("kafka-python not installed, using mock client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def produce(self, topic: str, key: str, value: Dict, headers: Dict = None) -> None:
        """Produce message to Kafka."""
        kafka_headers = [(k, v.encode('utf-8')) for k, v in (headers or {}).items()]
        future = self.producer.send(
            topic,
            key=key,
            value=value,
            headers=kafka_headers
        )
        future.get(timeout=10)  # Block until sent
        logger.info(f"Produced message to {topic}: {key}")

    def consume(self, topics: List[str], handler: Callable[[Dict], None]) -> None:
        """Consume messages from Kafka topics."""
        from kafka import KafkaConsumer

        consumer_config = {
            'bootstrap_servers': self.config.bootstrap_servers,
            'group_id': self.config.group_id,
            'client_id': f"{self.config.client_id}-consumer",
            'value_deserializer': lambda v: json.loads(v.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
            'auto_offset_reset': 'earliest'
        }

        if self.config.sasl_mechanism:
            consumer_config.update({
                'security_protocol': self.config.security_protocol,
                'sasl_mechanism': self.config.sasl_mechanism,
                'sasl_plain_username': self.config.sasl_username,
                'sasl_plain_password': self.config.sasl_password
            })

        self.consumer = KafkaConsumer(*topics, **consumer_config)

        for message in self.consumer:
            msg_dict = {
                'topic': message.topic,
                'key': message.key,
                'value': message.value,
                'headers': dict(message.headers) if message.headers else {},
                'timestamp': message.timestamp
            }
            handler(msg_dict)

    def close(self) -> None:
        """Close Kafka connections."""
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
        logger.info("Closed Kafka connections")


# =============================================================================
# Proof Streaming Service
# =============================================================================

class ProofStreamingService:
    """
    Service for streaming proofs to/from Kafka.

    Integrates with:
    - TOON serialization for efficient encoding
    - All 7 proof domains
    - Cross-domain proof linking
    """

    def __init__(self, config: KafkaConfig = None, use_mock: bool = True):
        self.config = config or KafkaConfig()
        self.encoder = TOONEncoder()

        if use_mock:
            self.client = MockKafkaClient(self.config)
        else:
            try:
                self.client = RealKafkaClient(self.config)
            except (ImportError, Exception):
                logger.warning("Falling back to MockKafkaClient")
                self.client = MockKafkaClient(self.config)

        self._event_handlers: Dict[str, List[Callable]] = {}

    def publish_proof(
        self,
        proof: UnifiedProof,
        event_type: ProofEventType = ProofEventType.CREATED
    ) -> ProofEvent:
        """Publish a proof to Kafka."""
        event = create_proof_event(proof, event_type, self.encoder)

        # Publish to domain-specific topic
        domain_topic = self.config.topics.get(proof.domain, self.config.topics["all"])
        kafka_msg = event.to_kafka_message()

        self.client.produce(
            topic=domain_topic,
            key=kafka_msg['key'],
            value=kafka_msg['value'],
            headers=kafka_msg['headers']
        )

        # Also publish to combined events topic
        self.client.produce(
            topic=self.config.topics["events"],
            key=kafka_msg['key'],
            value=kafka_msg['value'],
            headers=kafka_msg['headers']
        )

        logger.info(f"Published proof {proof.id} to {domain_topic}")
        return event

    def link_proofs(
        self,
        source_proof: UnifiedProof,
        target_proof: UnifiedProof,
        relationship: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Create a cross-domain link between proofs."""
        link_event = {
            'link_id': f"{source_proof.id}->{target_proof.id}",
            'source_id': source_proof.id,
            'source_domain': source_proof.domain,
            'target_id': target_proof.id,
            'target_domain': target_proof.domain,
            'relationship': relationship,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        }

        self.client.produce(
            topic=self.config.topics["links"],
            key=link_event['link_id'],
            value=link_event,
            headers={
                'source_domain': source_proof.domain,
                'target_domain': target_proof.domain,
                'relationship': relationship
            }
        )

        logger.info(f"Created link: {source_proof.id} --[{relationship}]--> {target_proof.id}")

    def subscribe(
        self,
        domains: List[str] = None,
        handler: Callable[[ProofEvent], None] = None
    ) -> None:
        """Subscribe to proof events."""
        if domains is None:
            topics = [self.config.topics["events"]]
        else:
            topics = [self.config.topics.get(d, self.config.topics["all"]) for d in domains]

        def message_handler(msg: Dict):
            if handler:
                # Reconstruct ProofEvent from message
                value = msg['value']
                event = ProofEvent(
                    event_id=value['event_id'],
                    event_type=ProofEventType(value['event_type']),
                    proof_id=value['proof_id'],
                    domain=value['domain'],
                    timestamp=datetime.fromisoformat(value['timestamp']),
                    payload=value['payload']
                )
                handler(event)

        self.client.consume(topics, message_handler)

    def close(self):
        """Close the streaming service."""
        self.client.close()


# =============================================================================
# N8n Integration
# =============================================================================

@dataclass
class N8nWebhookConfig:
    """Configuration for n8n webhook integration."""
    base_url: str = "http://localhost:5678"
    workflow_id: Optional[str] = None
    webhook_path: str = "/webhook/proof-events"


class N8nIntegration:
    """
    Integration with n8n workflows for proof processing.

    Supports:
    - Triggering n8n workflows on proof events
    - Receiving proof updates from n8n
    - Bi-directional sync with Kafka
    """

    def __init__(
        self,
        streaming_service: ProofStreamingService,
        n8n_config: N8nWebhookConfig = None
    ):
        self.streaming = streaming_service
        self.n8n_config = n8n_config or N8nWebhookConfig()

    def trigger_workflow(self, proof: UnifiedProof, workflow_action: str) -> Dict:
        """Trigger an n8n workflow for a proof."""
        import requests

        payload = {
            'action': workflow_action,
            'proof_id': proof.id,
            'domain': proof.domain,
            'title': proof.title,
            'status': proof.status.value,
            'toon_payload': self.streaming.encoder.encode(proof)
        }

        try:
            response = requests.post(
                f"{self.n8n_config.base_url}{self.n8n_config.webhook_path}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to trigger n8n workflow: {e}")
            return {'success': False, 'error': str(e)}

    def create_n8n_kafka_node_config(self) -> Dict:
        """Generate n8n Kafka node configuration for proof streaming."""
        return {
            'name': 'Proof Domains Kafka',
            'type': 'n8n-nodes-base.kafka',
            'parameters': {
                'topic': self.streaming.config.topics["events"],
                'brokers': self.streaming.config.bootstrap_servers,
                'clientId': self.streaming.config.client_id,
                'ssl': self.streaming.config.security_protocol != 'PLAINTEXT',
                'authentication': self.streaming.config.sasl_mechanism is not None
            },
            'credentials': {
                'kafka': {
                    'clientId': self.streaming.config.client_id,
                    'brokers': self.streaming.config.bootstrap_servers,
                    'ssl': self.streaming.config.security_protocol != 'PLAINTEXT',
                    'authentication': self.streaming.config.sasl_mechanism or 'none',
                    'username': self.streaming.config.sasl_username or '',
                    'password': self.streaming.config.sasl_password or ''
                }
            }
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_streaming_service(
    bootstrap_servers: str = "localhost:9092",
    use_mock: bool = True
) -> ProofStreamingService:
    """Create a proof streaming service."""
    config = KafkaConfig(bootstrap_servers=bootstrap_servers)
    return ProofStreamingService(config, use_mock=use_mock)


def get_domain_topic(domain: str, config: KafkaConfig = None) -> str:
    """Get Kafka topic for a domain."""
    config = config or KafkaConfig()
    return config.topics.get(domain, config.topics["all"])
