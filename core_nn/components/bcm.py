"""
Biological Core Memory (BCM) - Hippocampus-inspired temporal memory system.

The BCM maintains a fixed-size sliding window of recent representations,
selectively retaining only contextually salient events above a threshold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

from ..config.schema import BCMConfig


@dataclass
class MemorySlot:
    """Individual memory slot in the BCM."""
    embedding: torch.Tensor
    salience: float
    timestamp: int
    context_tags: Optional[Dict[str, Any]] = None


class SalienceGate(nn.Module):
    """Determines which memories are salient enough to retain."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute salience score for embedding."""
        return self.gate(embedding).squeeze(-1)


class MemoryUpdateGate(nn.Module):
    """Controls how memories are updated and integrated."""
    
    def __init__(self, embedding_dim: int, gate_type: str = "gru"):
        super().__init__()
        self.gate_type = gate_type
        
        if gate_type == "gru":
            self.update_gate = nn.GRUCell(embedding_dim, embedding_dim)
        elif gate_type == "lstm":
            self.update_gate = nn.LSTMCell(embedding_dim, embedding_dim)
        elif gate_type == "linear":
            self.update_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Update memory with new information."""
        if self.gate_type == "gru":
            return self.update_gate(current, previous)
        elif self.gate_type == "lstm":
            # For LSTM, we only use the hidden state
            hidden, _ = self.update_gate(current, (previous, torch.zeros_like(previous)))
            return hidden
        elif self.gate_type == "linear":
            combined = torch.cat([current, previous], dim=-1)
            return torch.tanh(self.update_gate(combined))


class BiologicalCoreMemory(nn.Module):
    """
    Biological Core Memory (BCM) - Hippocampus-inspired temporal memory.
    
    Maintains a fixed-size sliding window of embeddings, selectively retaining
    only contextually salient events above a threshold. Supports dynamic
    memory consolidation and retrieval.
    """
    
    def __init__(self, config: BCMConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.embedding_dim = config.embedding_dim
        self.salience_threshold = config.salience_threshold
        self.decay_rate = config.decay_rate
        
        # Memory storage
        self.memory_slots = []
        self.current_timestamp = 0
        
        # Neural components
        self.salience_gate = SalienceGate(config.embedding_dim)
        self.update_gate = MemoryUpdateGate(config.embedding_dim, config.update_gate_type)
        
        # Multi-head attention for memory retrieval
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.attention_heads,
            batch_first=True
        )
        
        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
    def forward(self, 
                input_embedding: torch.Tensor,
                query_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input through BCM and return retrieved memories.
        
        Args:
            input_embedding: Current input embedding [batch_size, embedding_dim]
            query_embedding: Optional query for memory retrieval
            
        Returns:
            retrieved_memory: Retrieved and consolidated memory
            memory_info: Dictionary with memory statistics
        """
        batch_size = input_embedding.size(0)
        
        # Compute salience for input
        salience_scores = self.salience_gate(input_embedding)
        
        # Store salient memories
        for i in range(batch_size):
            if salience_scores[i].item() > self.salience_threshold:
                self._store_memory(input_embedding[i], salience_scores[i].item())
        
        # Retrieve relevant memories
        if query_embedding is None:
            query_embedding = input_embedding
            
        retrieved_memory, attention_weights = self._retrieve_memories(query_embedding)
        
        # Apply memory decay
        self._apply_decay()
        
        # Increment timestamp
        self.current_timestamp += 1
        
        # Prepare memory info
        memory_info = {
            "num_stored_memories": len(self.memory_slots),
            "average_salience": np.mean([slot.salience for slot in self.memory_slots]) if self.memory_slots else 0.0,
            "attention_weights": attention_weights,
            "memory_utilization": len(self.memory_slots) / self.memory_size
        }
        
        return retrieved_memory, memory_info
    
    def _store_memory(self, embedding: torch.Tensor, salience: float):
        """Store a new memory slot."""
        memory_slot = MemorySlot(
            embedding=embedding.detach().clone(),
            salience=salience,
            timestamp=self.current_timestamp
        )
        
        # If memory is full, remove least salient memory
        if len(self.memory_slots) >= self.memory_size:
            # Find least salient memory
            min_salience_idx = min(range(len(self.memory_slots)), 
                                 key=lambda i: self.memory_slots[i].salience)
            
            # Only replace if new memory is more salient
            if salience > self.memory_slots[min_salience_idx].salience:
                self.memory_slots[min_salience_idx] = memory_slot
        else:
            self.memory_slots.append(memory_slot)
    
    def _retrieve_memories(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve memories using attention mechanism."""
        if not self.memory_slots:
            # Return zero tensor if no memories
            return torch.zeros_like(query), torch.zeros(query.size(0), 1)
        
        # Stack memory embeddings
        memory_embeddings = torch.stack([slot.embedding for slot in self.memory_slots])
        memory_embeddings = memory_embeddings.unsqueeze(0).repeat(query.size(0), 1, 1)
        
        # Apply attention
        retrieved, attention_weights = self.attention(
            query.unsqueeze(1),  # Add sequence dimension
            memory_embeddings,
            memory_embeddings
        )
        
        # Apply consolidation
        consolidated = self.consolidation_net(retrieved.squeeze(1))
        
        return consolidated, attention_weights.squeeze(1)
    
    def _apply_decay(self):
        """Apply temporal decay to memory salience."""
        for slot in self.memory_slots:
            slot.salience *= self.decay_rate
        
        # Remove memories below threshold
        self.memory_slots = [slot for slot in self.memory_slots 
                           if slot.salience > self.salience_threshold * 0.1]
    
    def remember_explicit(self, embedding: torch.Tensor, context: Dict[str, Any]):
        """Explicitly store a memory with high salience (user command)."""
        memory_slot = MemorySlot(
            embedding=embedding.detach().clone(),
            salience=1.0,  # Maximum salience for explicit memories
            timestamp=self.current_timestamp,
            context_tags=context
        )
        
        # Always store explicit memories, removing oldest if necessary
        if len(self.memory_slots) >= self.memory_size:
            # Remove oldest memory
            oldest_idx = min(range(len(self.memory_slots)), 
                           key=lambda i: self.memory_slots[i].timestamp)
            self.memory_slots[oldest_idx] = memory_slot
        else:
            self.memory_slots.append(memory_slot)
    
    def recall_by_context(self, context_query: Dict[str, Any], top_k: int = 5) -> List[MemorySlot]:
        """Recall memories by context tags."""
        matching_memories = []
        
        for slot in self.memory_slots:
            if slot.context_tags:
                # Simple context matching (can be made more sophisticated)
                match_score = 0
                for key, value in context_query.items():
                    if key in slot.context_tags and slot.context_tags[key] == value:
                        match_score += 1
                
                if match_score > 0:
                    matching_memories.append((slot, match_score))
        
        # Sort by match score and salience
        matching_memories.sort(key=lambda x: (x[1], x[0].salience), reverse=True)
        
        return [slot for slot, _ in matching_memories[:top_k]]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_slots:
            return {"num_memories": 0, "total_salience": 0.0, "average_age": 0.0}
        
        saliences = [slot.salience for slot in self.memory_slots]
        ages = [self.current_timestamp - slot.timestamp for slot in self.memory_slots]
        
        return {
            "num_memories": len(self.memory_slots),
            "total_salience": sum(saliences),
            "average_salience": np.mean(saliences),
            "max_salience": max(saliences),
            "min_salience": min(saliences),
            "average_age": np.mean(ages),
            "oldest_memory_age": max(ages),
            "memory_utilization": len(self.memory_slots) / self.memory_size
        }
