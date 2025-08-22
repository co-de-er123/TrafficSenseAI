""
AWS IoT Core configuration for TrafficSense AI.

Replace the placeholder values with your AWS IoT Core endpoint and certificate paths.
"""

# AWS IoT Core endpoint (without https://)
AWS_IOT_ENDPOINT = "YOUR_IOT_ENDPOINT.iot.REGION.amazonaws.com"

# Certificate paths (relative to the project root)
CERTIFICATE_PATHS = {
    'root_ca': "./certs/AmazonRootCA1.pem",
    'certificate': "./certs/device.pem.crt",
    'private_key': "./certs/private.pem.key"
}

# MQTT Topics
MQTT_TOPICS = {
    'detections': 'traffic/detections',
    'metrics': 'traffic/metrics',
    'commands': 'traffic/commands'
}

# Client ID (must be unique per device)
CLIENT_ID = "traffic-sense-ai-1"

# QoS level (0=at most once, 1=at least once, 2=exactly once)
QOS_LEVEL = 1

# Keep alive interval in seconds
KEEP_ALIVE = 60
