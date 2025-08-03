from typing import List
from embedding import vector_store
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_topic_history() -> List[str]:
    """Retrieve unique topics from ChromaDB."""
    try:
        results = vector_store.get()
        topics = set()
        for metadata in results.get("metadatas", []):
            topic = metadata.get("topic")
            if topic:
                topics.add(topic.lower())
        return sorted(list(topics))
    except Exception as e:
        logger.error(f"Failed to fetch topic history: {e}")
        return []




def disambiguate_topic(topic: str) -> str:
    """Add context to ambiguous topics for better search results."""
    topic = topic.strip().lower()
    academic_context = {
        "stack": "stack data structure",
        "agile": "agile software development",
        "benzene": "benzene chemistry",
        "attention mechanism": "attention mechanism neural networks",
        "newton third low of motion": "Newton's Third Law of Motion",
        "newton's third law": "Newton's Third Law of Motion"
    }
    return academic_context.get(topic, topic)