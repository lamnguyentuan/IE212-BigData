from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import os

BROKER = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "video_events")

def create_topic():
    print(f"Connecting to Kafka at {BROKER}...")
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=BROKER, 
            client_id='admin_client'
        )
        
        topic_list = [NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)]
        
        print(f"Creating topic: {TOPIC}")
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        print("Topic created successfully.")
        
    except TopicAlreadyExistsError:
        print("Topic already exists.")
    except Exception as e:
        print(f"Failed to create topic: {e}")
    finally:
        if 'admin_client' in locals():
            admin_client.close()

if __name__ == "__main__":
    create_topic()
