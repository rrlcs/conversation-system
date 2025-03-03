from conversation_agent import ConversationAgent
from services.vector_search import VectorSearchService
from utils.cache import VectorCache, BatchProcessor

__version__ = "0.1.0"

def initialize_system(
    pinecone_api_key: str,
    pinecone_env: str,
    index_name: str,
    together_api_key: str,
    cache_ttl: int = 3600,
    batch_size: int = 10
) -> ConversationAgent:
    """
    Initialize the conversation system with optimized components
    
    Args:
        pinecone_api_key: API key for Pinecone
        pinecone_env: Pinecone environment
        index_name: Name of the Pinecone index
        together_api_key: API key for Together AI
        cache_ttl: Time to live for cache entries (default: 1 hour)
        batch_size: Size of batches for processing (default: 10)
    
    Returns:
        Initialized ConversationAgent
    """
    return ConversationAgent(
        pinecone_api_key=pinecone_api_key,
        pinecone_env=pinecone_env,
        index_name=index_name,
        together_api_key=together_api_key
    ) 