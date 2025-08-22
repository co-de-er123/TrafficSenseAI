""
AWS IoT Core integration for TrafficSense AI.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import ssl
import paho.mqtt.client as mqtt
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AWSIoTConfig:
    """Configuration for AWS IoT Core connection."""
    endpoint: str  # Your AWS IoT endpoint (e.g., abc123-ats.iot.region.amazonaws.com)
    port: int = 8883
    client_id: str = "traffic-sense-ai"
    root_ca_path: str = "./certs/AmazonRootCA1.pem"
    cert_path: str = "./certs/device.pem.crt"
    private_key_path: str = "./certs/private.pem.key"
    topic: str = "traffic/vehicles"
    qos: int = 1

class MQTTClient:
    """MQTT client for AWS IoT Core communication."""
    
    def __init__(self, config: AWSIoTConfig):
        """Initialize the MQTT client with AWS IoT Core configuration."""
        self.config = config
        self.client = None
        self.connected = False
        self.on_message_callback = None
    
    def connect(self) -> bool:
        """Connect to AWS IoT Core."""
        try:
            # Configure client
            self.client = AWSIoTMQTTClient(self.config.client_id)
            self.client.configureEndpoint(self.config.endpoint, self.config.port)
            self.client.configureCredentials(
                self.config.root_ca_path,
                self.config.private_key_path,
                self.config.cert_path
            )
            
            # Configure connection settings
            self.client.configureAutoReconnectBackoffTime(1, 32, 20)
            self.client.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
            self.client.configureDrainingFrequency(2)  # Draining: 2 Hz
            self.client.configureConnectDisconnectTimeout(10)  # 10 sec
            self.client.configureMQTTOperationTimeout(5)  # 5 sec
            
            # Connect to AWS IoT
            self.client.connect()
            self.connected = True
            logger.info("Connected to AWS IoT Core")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to AWS IoT: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from AWS IoT Core."""
        if self.client and self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from AWS IoT Core")
    
    def publish(self, payload: Dict[str, Any]) -> bool:
        """Publish a message to the configured topic.
        
        Args:
            payload: Dictionary containing the message payload
            
        Returns:
            bool: True if the message was published successfully, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to AWS IoT Core")
            return False
            
        try:
            self.client.publish(
                self.config.topic,
                json.dumps(payload),
                self.config.qos
            )
            logger.debug(f"Published message: {payload}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}")
            return False
    
    def subscribe(self, topic: Optional[str] = None, callback: Optional[Callable] = None) -> bool:
        """Subscribe to a topic and set a callback for incoming messages.
        
        Args:
            topic: Topic to subscribe to (defaults to the configured topic)
            callback: Function to call when a message is received
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to AWS IoT Core")
            return False
            
        topic = topic or self.config.topic
        self.on_message_callback = callback
        
        try:
            def on_message(client, userdata, message):
                try:
                    payload = json.loads(message.payload)
                    if self.on_message_callback:
                        self.on_message_callback(payload)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message: {message.payload}")
            
            self.client.subscribe(topic, self.config.qos, on_message)
            logger.info(f"Subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {str(e)}")
            return False

class TrafficDataPublisher:
    """Handles publishing traffic data to AWS IoT Core."""
    
    def __init__(self, mqtt_client: MQTTClient):
        """Initialize with an MQTT client."""
        self.mqtt_client = mqtt_client
    
    def publish_detection(self, detections: list, frame_timestamp: float) -> bool:
        """Publish vehicle detection data.
        
        Args:
            detections: List of detection dictionaries
            frame_timestamp: Timestamp of the frame
            
        Returns:
            bool: True if the data was published successfully
        """
        payload = {
            "timestamp": frame_timestamp,
            "detections": detections,
            "count": len(detections)
        }
        return self.mqtt_client.publish(payload)
    
    def publish_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Publish system metrics.
        
        Args:
            metrics: Dictionary containing system metrics
            
        Returns:
            bool: True if the data was published successfully
        """
        payload = {
            "timestamp": time.time(),
            "type": "metrics",
            "data": metrics
        }
        return self.mqtt_client.publish(payload)
