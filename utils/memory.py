from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, List
from langchain.schema import BaseMessage

class ScalableConversationMemory(ConversationBufferMemory):
    """Extended ConversationBufferMemory with scalability features."""
    
    def __init__(
        self,
        max_entries: int = 1000,
        cleanup_threshold: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._max_entries = max_entries
        self._cleanup_threshold = cleanup_threshold
        
    @property
    def max_entries(self) -> int:
        """Maximum number of entries to store in memory."""
        return self._max_entries
    
    @property
    def cleanup_threshold(self) -> float:
        """Threshold at which to clean up memory (as a fraction of max_entries)."""
        return self._cleanup_threshold
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context with automatic cleanup if needed."""
        current_size = len(self.chat_memory.messages)
        
        if current_size >= self.max_entries * self.cleanup_threshold:
            # Remove oldest entries to make space
            num_to_remove = int(current_size * 0.2)  # Remove 20% of entries
            self.chat_memory.messages = self.chat_memory.messages[num_to_remove:]
        
        super().save_context(inputs, outputs)
    
    def get_message_count(self) -> int:
        """Get the current number of messages in memory."""
        return len(self.chat_memory.messages) 