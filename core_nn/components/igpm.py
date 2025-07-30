"""
Instruction-Guided Plasticity Module (IGPM) - Meta-learning without global weight updates.

The IGPM learns from user input using meta-updates without modifying global weights.
"Plastic slots" store pattern modifications tied to instruction-based usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque

from ..config.schema import IGPMConfig


@dataclass
class EpisodicMemory:
    """Individual episodic memory entry."""
    instruction: str
    context_embedding: torch.Tensor
    response_embedding: torch.Tensor
    timestamp: int
    usage_count: int = 0
    success_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class PlasticSlot(nn.Module):
    """Individual plastic memory slot for fast adaptation."""
    
    def __init__(self, embedding_dim: int, instruction_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.instruction_dim = instruction_dim
        
        # Fast weight matrices
        self.fast_weights = nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.fast_bias = nn.Parameter(torch.zeros(embedding_dim))
        
        # Instruction conditioning
        self.instruction_gate = nn.Sequential(
            nn.Linear(instruction_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        # Plasticity control
        self.plasticity_gate = nn.Sequential(
            nn.Linear(embedding_dim + instruction_dim, 1),
            nn.Sigmoid()
        )
        
        # Usage tracking
        self.register_buffer('usage_count', torch.zeros(1))
        self.register_buffer('last_update', torch.zeros(1))
        
    def forward(self, 
                x: torch.Tensor, 
                instruction_embedding: torch.Tensor) -> torch.Tensor:
        """Apply plastic transformation to input."""
        # Compute plasticity gate
        combined = torch.cat([x, instruction_embedding], dim=-1)
        plasticity = self.plasticity_gate(combined)
        
        # Apply fast weights
        transformed = torch.matmul(x, self.fast_weights) + self.fast_bias
        
        # Instruction-gated modulation
        gate = self.instruction_gate(instruction_embedding)
        modulated = transformed * gate
        
        # Plasticity-gated output
        output = x + plasticity * modulated
        
        return output
    
    def update_fast_weights(self, 
                           gradient: torch.Tensor, 
                           learning_rate: float,
                           decay_rate: float):
        """Update fast weights with gradient and decay."""
        with torch.no_grad():
            # Apply decay
            self.fast_weights.mul_(decay_rate)
            self.fast_bias.mul_(decay_rate)
            
            # Apply gradient update
            if gradient.dim() == 2:
                self.fast_weights.add_(gradient, alpha=learning_rate)
            else:
                self.fast_bias.add_(gradient, alpha=learning_rate)
            
            # Update usage tracking
            self.usage_count.add_(1)
            self.last_update.fill_(torch.tensor(0.0))  # Reset to current time


class InstructionEncoder(nn.Module):
    """Encodes natural language instructions into embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, instruction_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, instruction_dim // 2, 
                              batch_first=True, bidirectional=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, instruction_tokens: torch.Tensor) -> torch.Tensor:
        """Encode instruction tokens to embedding."""
        # Embed tokens
        embedded = self.embedding(instruction_tokens)
        
        # LSTM encoding
        encoded, _ = self.encoder(embedded)
        
        # Global average pooling
        pooled = self.pooling(encoded.transpose(1, 2)).squeeze(-1)
        
        return pooled


class MetaLearner(nn.Module):
    """Meta-learning component for fast adaptation."""
    
    def __init__(self, embedding_dim: int, instruction_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.instruction_dim = instruction_dim
        
        # Meta-learning network
        self.meta_net = nn.Sequential(
            nn.Linear(embedding_dim + instruction_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim * embedding_dim + embedding_dim)
        )
        
    def forward(self, 
                context: torch.Tensor, 
                instruction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fast weight updates from context and instruction."""
        # Combine context and instruction
        combined = torch.cat([context, instruction], dim=-1)
        
        # Generate weight updates
        updates = self.meta_net(combined)
        
        # Split into weight matrix and bias updates
        weight_updates = updates[:, :-self.embedding_dim].view(-1, self.embedding_dim, self.embedding_dim)
        bias_updates = updates[:, -self.embedding_dim:]
        
        return weight_updates, bias_updates


class InstructionGuidedPlasticityModule(nn.Module):
    """
    Instruction-Guided Plasticity Module (IGPM) - Meta-learning without global updates.
    
    Learns from user input using meta-updates without modifying global weights.
    "Plastic slots" store pattern modifications tied to instruction-based usage.
    """
    
    def __init__(self, config: IGPMConfig, vocab_size: int = 50000, embedding_dim: int = 768):
        super().__init__()
        self.config = config
        self.plastic_slots = config.plastic_slots
        self.embedding_dim = embedding_dim  # Use the actual embedding dimension from input
        self.instruction_embedding_dim = config.instruction_embedding_dim
        self.plasticity_threshold = config.plasticity_threshold

        # Instruction encoder
        self.instruction_encoder = InstructionEncoder(
            vocab_size, self.embedding_dim, config.instruction_embedding_dim
        )

        # Plastic slots
        self.slots = nn.ModuleList([
            PlasticSlot(self.embedding_dim, config.instruction_embedding_dim)
            for _ in range(config.plastic_slots)
        ])

        # Meta-learner
        self.meta_learner = MetaLearner(self.embedding_dim, config.instruction_embedding_dim)
        
        # Episodic memory
        self.episodic_memories = deque(maxlen=config.max_episodic_memories)
        self.memory_index = 0
        
        # Slot allocation tracking
        self.register_buffer('slot_usage', torch.zeros(config.plastic_slots))
        self.register_buffer('slot_success', torch.zeros(config.plastic_slots))
        
    def forward(self, 
                x: torch.Tensor,
                instruction: Optional[str] = None,
                instruction_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input through IGPM with optional instruction.
        
        Args:
            x: Input tensor [batch_size, embedding_dim]
            instruction: Natural language instruction
            instruction_tokens: Tokenized instruction [batch_size, seq_len]
            
        Returns:
            output: Processed output [batch_size, embedding_dim]
            info: Processing information
        """
        batch_size = x.size(0)
        
        # Encode instruction if provided
        if instruction_tokens is not None:
            instruction_embedding = self.instruction_encoder(instruction_tokens)
        elif instruction is not None:
            # Simple tokenization (in practice, use proper tokenizer)
            instruction_tokens = self._simple_tokenize(instruction, batch_size)
            instruction_embedding = self.instruction_encoder(instruction_tokens)
        else:
            # Default neutral instruction
            instruction_embedding = torch.zeros(batch_size, self.config.instruction_embedding_dim, 
                                              device=x.device)
        
        # Find relevant plastic slots
        relevant_slots = self._find_relevant_slots(x, instruction_embedding)
        
        # Apply plastic transformations
        output = x
        slot_activations = []
        
        for slot_idx in relevant_slots:
            slot = self.slots[slot_idx]
            slot_output = slot(output, instruction_embedding)
            
            # Compute activation strength
            activation = torch.norm(slot_output - output, dim=-1).mean().item()
            slot_activations.append(activation)
            
            # Update output if activation is strong enough
            if activation > self.plasticity_threshold:
                output = slot_output
                self.slot_usage[slot_idx] += 1
        
        # Store episodic memory if instruction provided
        if instruction is not None:
            self._store_episodic_memory(instruction, x, output)
        
        # Prepare info
        info = {
            "relevant_slots": relevant_slots,
            "slot_activations": slot_activations,
            "instruction_norm": torch.norm(instruction_embedding).item(),
            "plasticity_applied": len([a for a in slot_activations if a > self.plasticity_threshold])
        }
        
        return output, info
    
    def learn_from_instruction(self, 
                              instruction: str,
                              context: torch.Tensor,
                              target: torch.Tensor) -> Dict[str, Any]:
        """Learn from instruction-context-target triplet."""
        # Encode instruction
        instruction_tokens = self._simple_tokenize(instruction, context.size(0))
        instruction_embedding = self.instruction_encoder(instruction_tokens)
        
        # Generate meta-updates
        weight_updates, bias_updates = self.meta_learner(context, instruction_embedding)
        
        # Find or allocate slot
        slot_idx = self._allocate_slot(instruction_embedding)
        slot = self.slots[slot_idx]
        
        # Compute loss for meta-learning
        predicted = slot(context, instruction_embedding)
        loss = F.mse_loss(predicted, target)
        
        # Compute gradients for fast weights
        weight_grad = torch.autograd.grad(loss, slot.fast_weights, 
                                        retain_graph=True, create_graph=False)[0]
        bias_grad = torch.autograd.grad(loss, slot.fast_bias, 
                                      retain_graph=True, create_graph=False)[0]
        
        # Update fast weights
        slot.update_fast_weights(
            weight_grad, 
            self.config.meta_learning_rate,
            self.config.fast_weight_decay
        )
        
        # Store episodic memory
        self._store_episodic_memory(instruction, context, target)
        
        return {
            "slot_used": slot_idx,
            "loss": loss.item(),
            "weight_update_norm": torch.norm(weight_grad).item(),
            "bias_update_norm": torch.norm(bias_grad).item()
        }
    
    def remember_explicit(self, instruction: str, context: torch.Tensor):
        """Explicitly remember instruction-context pair."""
        # Encode instruction
        instruction_tokens = self._simple_tokenize(instruction, context.size(0))
        instruction_embedding = self.instruction_encoder(instruction_tokens)
        
        # Store in episodic memory with high importance
        memory = EpisodicMemory(
            instruction=instruction,
            context_embedding=context.detach().clone(),
            response_embedding=context.detach().clone(),  # Same as context for explicit memory
            timestamp=self.memory_index,
            usage_count=0,
            success_score=1.0,  # High success for explicit memories
            metadata={"explicit": True}
        )
        
        self.episodic_memories.append(memory)
        self.memory_index += 1
        
        # Allocate dedicated slot
        slot_idx = self._allocate_slot(instruction_embedding, force_new=True)
        
        return {"slot_allocated": slot_idx, "memory_stored": True}
    
    def recall_by_instruction(self, instruction: str, top_k: int = 5) -> List[EpisodicMemory]:
        """Recall episodic memories by instruction similarity."""
        if not self.episodic_memories:
            return []
        
        # Encode query instruction
        instruction_tokens = self._simple_tokenize(instruction, 1)
        query_embedding = self.instruction_encoder(instruction_tokens)
        
        # Compute similarities
        similarities = []
        for memory in self.episodic_memories:
            # Simple similarity based on instruction text (can be improved)
            similarity = self._compute_instruction_similarity(instruction, memory.instruction)
            similarities.append((memory, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, _ in similarities[:top_k]]
    
    def _find_relevant_slots(self, 
                           context: torch.Tensor, 
                           instruction: torch.Tensor) -> List[int]:
        """Find plastic slots relevant to current context and instruction."""
        # Simple relevance based on usage and recent activity
        relevance_scores = []
        
        for i, slot in enumerate(self.slots):
            # Usage-based relevance
            usage_score = self.slot_usage[i].item()
            
            # Recency-based relevance (simplified)
            recency_score = 1.0 / (slot.last_update.item() + 1.0)
            
            # Combined relevance
            relevance = usage_score * 0.7 + recency_score * 0.3
            relevance_scores.append((i, relevance))
        
        # Sort by relevance and return top slots
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in relevance_scores[:min(3, len(relevance_scores))]]
    
    def _allocate_slot(self, 
                      instruction_embedding: torch.Tensor, 
                      force_new: bool = False) -> int:
        """Allocate a plastic slot for new learning."""
        if force_new:
            # Find least used slot
            return torch.argmin(self.slot_usage).item()
        
        # Find slot with lowest usage
        min_usage_idx = torch.argmin(self.slot_usage).item()
        
        # If usage is very low, reuse the slot
        if self.slot_usage[min_usage_idx] < 5:
            return min_usage_idx
        
        # Otherwise, find least recently used
        last_updates = torch.tensor([slot.last_update.item() for slot in self.slots])
        return torch.argmax(last_updates).item()
    
    def _store_episodic_memory(self, 
                              instruction: str,
                              context: torch.Tensor,
                              response: torch.Tensor):
        """Store new episodic memory."""
        memory = EpisodicMemory(
            instruction=instruction,
            context_embedding=context.detach().clone(),
            response_embedding=response.detach().clone(),
            timestamp=self.memory_index,
            usage_count=0,
            success_score=0.5  # Neutral initial score
        )
        
        self.episodic_memories.append(memory)
        self.memory_index += 1
    
    def _simple_tokenize(self, text: str, batch_size: int) -> torch.Tensor:
        """Simple tokenization (replace with proper tokenizer in practice)."""
        # Convert text to token IDs (simplified)
        # Ensure token IDs are within vocabulary range (vocab_size from instruction encoder)
        vocab_size = self.instruction_encoder.embedding.num_embeddings
        tokens = [min(ord(c) % vocab_size, vocab_size - 1) for c in text[:50]]  # Limit length and vocab range
        tokens = tokens + [0] * (50 - len(tokens))  # Pad

        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    
    def _compute_instruction_similarity(self, inst1: str, inst2: str) -> float:
        """Compute similarity between instructions (simplified)."""
        # Simple word overlap similarity
        words1 = set(inst1.lower().split())
        words2 = set(inst2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_plasticity_stats(self) -> Dict[str, Any]:
        """Get comprehensive plasticity statistics."""
        return {
            "total_slots": self.plastic_slots,
            "active_slots": (self.slot_usage > 0).sum().item(),
            "total_usage": self.slot_usage.sum().item(),
            "average_usage": self.slot_usage.mean().item(),
            "episodic_memories": len(self.episodic_memories),
            "memory_capacity": self.config.max_episodic_memories
        }
