from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from utils.cache import VectorCache, BatchProcessor
import asyncio

class VectorSearchService:
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_env: str,
        index_name: str,
        embedding_model: str = 'all-MiniLM-L6-v2',
        cache_ttl: int = 3600,
        batch_size: int = 10,
        max_retries: int = 3
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        # Initialize caching and batching
        self.cache = VectorCache(ttl=cache_ttl)
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.max_retries = max_retries
        
        # Pre-warm the model
        self._warm_up_model()
    
    def _warm_up_model(self) -> None:
        """Warm up the embedding model to reduce initial latency"""
        self.embedding_model.encode(["warm up query"])
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available"""
        cache_key = f"emb_{hash(text)}"
        return self.cache.get(cache_key)
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for future use"""
        cache_key = f"emb_{hash(text)}"
        self.cache.set(cache_key, embedding)
    
    def encode_text(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available"""
        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            return cached_embedding
        
        embedding = self.embedding_model.encode(text)
        self._cache_embedding(text, embedding.tolist())
        return embedding.tolist()
    
    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts efficiently"""
        # Check cache first
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        if texts_to_encode:
            # Encode new texts in a batch
            new_embeddings = self.embedding_model.encode(texts_to_encode)
            
            # Cache new embeddings
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self._cache_embedding(text, embedding.tolist())
            
            # Merge cached and new embeddings
            for idx, embedding in zip(text_indices, new_embeddings):
                embeddings.insert(idx, embedding.tolist())
        
        return embeddings
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimized semantic search with caching and filtering
        """
        for attempt in range(self.max_retries):
            try:
                # Generate query embedding
                query_embedding = self.encode_text(query)
                
                # Search in vector store with optional filters
                search_results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                
                return self._process_search_results(search_results)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                continue
    
    def batch_upsert(self, items: List[Dict[str, Any]]) -> None:
        """
        Efficiently upsert multiple items in batches
        """
        # Extract texts for batch encoding
        texts = [item['text'] for item in items]
        embeddings = self.batch_encode(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for item, embedding in zip(items, embeddings):
            vectors.append({
                'id': item['id'],
                'values': embedding,
                'metadata': {
                    'time': item.get('time'),
                    'user': item.get('user'),
                    'ai': item.get('ai'),
                    'text': item.get('text')
                }
            })
        
        # Batch upsert to Pinecone
        self.index.upsert(vectors=vectors)
    
    def _process_search_results(self, results: Any) -> List[Dict[str, Any]]:
        """Process and format search results"""
        processed_results = []
        for match in results.matches:
            if match.score > 0.2:  # Configurable threshold
                processed_results.append({
                    'score': match.score,
                    'metadata': match.metadata
                })
        return processed_results

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts."""
        return self.batch_encode(texts)

    async def batch_upsert_async(self, items: List[Dict[str, Any]]) -> None:
        """
        Asynchronously upsert multiple items in batches
        """
        # Run batch_upsert in executor to make it async
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.batch_upsert,
            items
        ) 