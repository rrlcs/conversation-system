from typing import List, Any, Callable, TypeVar, Generic
import asyncio
from collections import deque
import time
from config import ScalabilityConfig

T = TypeVar('T')
R = TypeVar('R')

class BatchProcessor(Generic[T, R]):
    def __init__(
        self,
        process_func: Callable[[List[T]], List[R]],
        min_batch_size: int = ScalabilityConfig.MIN_BATCH_SIZE,
        max_batch_size: int = ScalabilityConfig.MAX_BATCH_SIZE,
        batch_timeout: float = ScalabilityConfig.BATCH_TIMEOUT
    ):
        self.process_func = process_func
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue: deque = deque()
        self.results: dict = {}
        self.batch_lock = asyncio.Lock()
        self.last_process_time = time.time()
    
    async def add_item(self, item: T) -> R:
        """Add an item to the batch queue and wait for its result."""
        async with self.batch_lock:
            # Generate unique ID for this item
            item_id = id(item)
            self.queue.append((item_id, item))
            
            # Process batch if conditions are met
            current_time = time.time()
            should_process = (
                len(self.queue) >= self.max_batch_size or
                (len(self.queue) >= self.min_batch_size and
                 current_time - self.last_process_time >= self.batch_timeout)
            )
            
            if should_process:
                await self._process_batch()
        
        # Wait for result
        while item_id not in self.results:
            await asyncio.sleep(0.01)
        
        result = self.results.pop(item_id)
        return result
    
    async def _process_batch(self) -> None:
        """Process the current batch of items."""
        if not self.queue:
            return
        
        # Get items from queue
        batch_size = min(len(self.queue), self.max_batch_size)
        batch_items = []
        batch_ids = []
        
        for _ in range(batch_size):
            item_id, item = self.queue.popleft()
            batch_items.append(item)
            batch_ids.append(item_id)
        
        # Process batch
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.process_func, batch_items
            )
            
            # Store results
            for item_id, result in zip(batch_ids, results):
                self.results[item_id] = result
        except Exception as e:
            # Handle errors by returning None for all items in batch
            for item_id in batch_ids:
                self.results[item_id] = None
            print(f"Batch processing error: {e}")
        
        self.last_process_time = time.time()
    
    async def process_remaining(self) -> None:
        """Process any remaining items in the queue."""
        async with self.batch_lock:
            if self.queue:
                await self._process_batch()

class BatchRequestHandler:
    def __init__(self):
        self.processors: dict = {}
    
    def get_processor(
        self,
        name: str,
        process_func: Callable[[List[T]], List[R]]
    ) -> BatchProcessor[T, R]:
        """Get or create a batch processor for the given name."""
        if name not in self.processors:
            self.processors[name] = BatchProcessor(process_func)
        return self.processors[name]
    
    async def cleanup(self) -> None:
        """Process remaining items in all processors."""
        for processor in self.processors.values():
            await processor.process_remaining() 