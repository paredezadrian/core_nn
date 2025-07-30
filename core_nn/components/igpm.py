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
    """Individual plastic memory slot with gradient-based fast weights."""

    def __init__(self, embedding_dim: int, instruction_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.instruction_dim = instruction_dim

        # Fast weight matrices - initialize with small random values instead of zeros
        self.fast_weights = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
        self.fast_bias = nn.Parameter(torch.randn(embedding_dim) * 0.01)

        # Gradient accumulation for plasticity
        self.register_buffer('weight_gradients', torch.zeros_like(self.fast_weights))
        self.register_buffer('bias_gradients', torch.zeros_like(self.fast_bias))

        # Instruction conditioning - improved initialization
        self.instruction_gate = nn.Sequential(
            nn.Linear(instruction_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh()  # Changed from Sigmoid for better gradient flow
        )

        # Plasticity control - enhanced with better initialization
        self.plasticity_gate = nn.Sequential(
            nn.Linear(embedding_dim + instruction_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )

        # Enhanced usage tracking
        self.register_buffer('usage_count', torch.zeros(1))
        self.register_buffer('last_update', torch.zeros(1))
        self.register_buffer('plasticity_strength', torch.ones(1) * 0.1)  # Adaptive plasticity strength

        # Context-dependent plasticity parameters
        self.register_buffer('context_type', torch.zeros(1))  # 0=general, 1=memory, 2=attention, 3=suppression
        self.register_buffer('adaptation_rate', torch.ones(1) * 1.0)  # Context-specific adaptation rate

        # Neuromodulation-inspired plasticity
        self.register_buffer('dopamine_level', torch.ones(1) * 0.5)  # Reward/motivation signal
        self.register_buffer('acetylcholine_level', torch.ones(1) * 0.5)  # Attention/learning signal
        self.register_buffer('norepinephrine_level', torch.ones(1) * 0.5)  # Arousal/stress signal
        self.register_buffer('serotonin_level', torch.ones(1) * 0.5)  # Mood/stability signal
        
    def forward(self,
                x: torch.Tensor,
                instruction_embedding: torch.Tensor) -> torch.Tensor:
        """Apply context-dependent gradient-based plastic transformation to input."""
        # Compute plasticity gate with enhanced strength
        combined = torch.cat([x, instruction_embedding], dim=-1)
        plasticity = self.plasticity_gate(combined) * self.plasticity_strength

        # Apply fast weights with gradient-informed updates
        transformed = torch.matmul(x, self.fast_weights) + self.fast_bias

        # Instruction-gated modulation with improved activation
        gate = self.instruction_gate(instruction_embedding)
        modulated = transformed * gate

        # Context-dependent plasticity modulation
        context_modulation = self._get_context_modulation(instruction_embedding)

        # Neuromodulation-inspired plasticity enhancement
        neuromod_factor = self._compute_neuromodulation_factor()

        # Enhanced plasticity-gated output with context-dependent and neuromodulated strength
        residual_strength = 0.5  # Reduce residual connection strength
        plasticity_strength = 2.0 * context_modulation * self.adaptation_rate.item() * neuromod_factor
        output = x * residual_strength + plasticity * modulated * plasticity_strength

        return output

    def _get_context_modulation(self, instruction_embedding: torch.Tensor) -> float:
        """Get context-dependent modulation factor based on instruction type."""
        # Simple heuristic based on instruction embedding characteristics
        embedding_norm = torch.norm(instruction_embedding).item()
        embedding_mean = torch.mean(instruction_embedding).item()

        # Classify instruction type based on embedding characteristics
        if embedding_mean > 0.1:  # High positive mean suggests amplification/memory instructions
            self.context_type.fill_(1)  # Memory context
            self.adaptation_rate.fill_(1.5)  # Higher adaptation for memory tasks
            return 1.5
        elif embedding_mean < -0.1:  # Negative mean suggests suppression instructions
            self.context_type.fill_(3)  # Suppression context
            self.adaptation_rate.fill_(0.8)  # Lower adaptation for suppression
            return 0.8
        elif embedding_norm > 1.0:  # High norm suggests attention/focus instructions
            self.context_type.fill_(2)  # Attention context
            self.adaptation_rate.fill_(1.2)  # Moderate adaptation for attention
            return 1.2
        else:  # General instructions
            self.context_type.fill_(0)  # General context
            self.adaptation_rate.fill_(1.0)  # Standard adaptation
            return 1.0

    def _compute_neuromodulation_factor(self) -> float:
        """Compute neuromodulation factor based on neurotransmitter levels."""
        # Dopamine enhances learning when reward is expected
        dopamine_effect = 0.5 + self.dopamine_level.item() * 0.5

        # Acetylcholine enhances attention and learning
        acetylcholine_effect = 0.5 + self.acetylcholine_level.item() * 0.8

        # Norepinephrine modulates arousal (too high or low reduces plasticity)
        norepinephrine_effect = 1.0 - abs(self.norepinephrine_level.item() - 0.5) * 0.4

        # Serotonin provides stability (moderate levels are best)
        serotonin_effect = 1.0 - abs(self.serotonin_level.item() - 0.5) * 0.3

        # Combine effects multiplicatively
        combined_effect = dopamine_effect * acetylcholine_effect * norepinephrine_effect * serotonin_effect

        return max(0.1, min(2.0, combined_effect))  # Clamp to reasonable range

    def update_neuromodulation(self,
                              reward_signal: float = 0.0,
                              attention_demand: float = 0.0,
                              stress_level: float = 0.0,
                              stability_need: float = 0.0):
        """Update neuromodulator levels based on environmental signals."""
        with torch.no_grad():
            # Update dopamine based on reward signal
            self.dopamine_level.mul_(0.9).add_(reward_signal * 0.1)
            self.dopamine_level.clamp_(0.0, 1.0)

            # Update acetylcholine based on attention demand
            self.acetylcholine_level.mul_(0.9).add_(attention_demand * 0.1)
            self.acetylcholine_level.clamp_(0.0, 1.0)

            # Update norepinephrine based on stress level
            self.norepinephrine_level.mul_(0.9).add_(stress_level * 0.1)
            self.norepinephrine_level.clamp_(0.0, 1.0)

            # Update serotonin based on stability need
            self.serotonin_level.mul_(0.9).add_(stability_need * 0.1)
            self.serotonin_level.clamp_(0.0, 1.0)
    
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
            self.last_update.fill_(0.0)  # Reset to current time

    def update_with_gradients(self,
                             weight_grad: torch.Tensor,
                             bias_grad: torch.Tensor,
                             learning_rate: float,
                             decay_rate: float):
        """Enhanced gradient-based update with momentum and adaptive learning."""
        with torch.no_grad():
            # Accumulate gradients with momentum
            momentum = 0.9
            self.weight_gradients.mul_(momentum).add_(weight_grad, alpha=1-momentum)
            self.bias_gradients.mul_(momentum).add_(bias_grad, alpha=1-momentum)

            # Apply decay
            self.fast_weights.mul_(decay_rate)
            self.fast_bias.mul_(decay_rate)

            # Adaptive learning rate based on gradient magnitude
            grad_norm = torch.norm(self.weight_gradients).item()
            adaptive_lr = learning_rate * max(0.1, min(2.0, grad_norm))

            # Apply gradient updates
            self.fast_weights.add_(self.weight_gradients, alpha=adaptive_lr)
            self.fast_bias.add_(self.bias_gradients, alpha=adaptive_lr)

            # Update plasticity strength based on gradient activity
            self.plasticity_strength.mul_(0.99).add_(grad_norm * 0.01)
            self.plasticity_strength.clamp_(0.01, 1.0)

            # Update usage tracking
            self.usage_count.add_(1)
            self.last_update.fill_(0.0)


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
    """MAML-style meta-learning component for fast adaptation."""

    def __init__(self, embedding_dim: int, instruction_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.instruction_dim = instruction_dim

        # Enhanced meta-learning network with MAML-style architecture
        context_encoded_dim = embedding_dim // 4  # Output of context encoder
        input_dim = context_encoded_dim + instruction_dim
        hidden_dim = embedding_dim * 2
        output_dim = embedding_dim * embedding_dim + embedding_dim

        self.meta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Bounded outputs for stability
        )

        # MAML adaptation parameters
        self.adaptation_steps = 3
        self.inner_lr = 0.01

        # Context encoder for better meta-learning
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        
    def forward(self,
                context: torch.Tensor,
                instruction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate MAML-style fast weight updates from context and instruction."""
        # Encode context for better meta-learning
        context_encoded = self.context_encoder(context)

        # Combine encoded context and instruction
        combined = torch.cat([context_encoded, instruction], dim=-1)

        # Generate weight updates through meta-network
        updates = self.meta_net(combined)

        # Split into weight matrix and bias updates
        weight_updates = updates[:, :-self.embedding_dim].view(-1, self.embedding_dim, self.embedding_dim)
        bias_updates = updates[:, -self.embedding_dim:]

        return weight_updates, bias_updates

    def maml_adaptation(self,
                       context: torch.Tensor,
                       instruction: torch.Tensor,
                       target: torch.Tensor,
                       slot: 'PlasticSlot') -> Dict[str, float]:
        """Perform MAML-style multi-step adaptation."""
        adaptation_losses = []

        # Store original weights
        original_weights = slot.fast_weights.clone()
        original_bias = slot.fast_bias.clone()

        # Multi-step adaptation
        for step in range(self.adaptation_steps):
            # Forward pass
            predicted = slot(context, instruction)
            loss = F.mse_loss(predicted, target)
            adaptation_losses.append(loss.item())

            # Compute gradients
            if slot.fast_weights.grad is not None:
                slot.fast_weights.grad.zero_()
            if slot.fast_bias.grad is not None:
                slot.fast_bias.grad.zero_()

            loss.backward(retain_graph=True)

            # Inner loop update
            with torch.no_grad():
                if slot.fast_weights.grad is not None:
                    slot.fast_weights.sub_(slot.fast_weights.grad, alpha=self.inner_lr)
                if slot.fast_bias.grad is not None:
                    slot.fast_bias.sub_(slot.fast_bias.grad, alpha=self.inner_lr)

        return {
            "adaptation_steps": self.adaptation_steps,
            "final_loss": adaptation_losses[-1] if adaptation_losses else 0.0,
            "loss_reduction": adaptation_losses[0] - adaptation_losses[-1] if len(adaptation_losses) > 1 else 0.0
        }


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
        
        # Apply plastic transformations with improved activation logic
        output = x
        slot_activations = []
        total_plasticity_effect = 0.0

        for slot_idx in relevant_slots:
            slot = self.slots[slot_idx]
            slot_output = slot(x, instruction_embedding)  # Always use original input

            # Compute activation strength relative to input (not current output)
            activation = torch.norm(slot_output - x, dim=-1).mean().item()
            slot_activations.append(activation)

            # Use a much lower threshold and blend outputs instead of hard switching
            if activation > 0.001:  # Very low threshold
                # Blend the slot output with current output
                blend_weight = min(1.0, activation / 0.1)  # Scale activation to blend weight
                output = output * (1 - blend_weight) + slot_output * blend_weight
                self.slot_usage[slot_idx] += 1
                total_plasticity_effect += activation
        
        # Store episodic memory if instruction provided
        if instruction is not None:
            self._store_episodic_memory(instruction, x, output)
        
        # Prepare info
        info = {
            "relevant_slots": relevant_slots,
            "slot_activations": slot_activations,
            "total_plasticity_effect": total_plasticity_effect,
            "instruction_norm": torch.norm(instruction_embedding).item(),
            "plasticity_applied": len([a for a in slot_activations if a > 0.001])  # Use new threshold
        }
        
        return output, info
    
    def learn_from_instruction(self,
                              instruction: str,
                              context: torch.Tensor,
                              target: torch.Tensor) -> Dict[str, Any]:
        """Enhanced learning from instruction-context-target triplet with gradient-based plasticity."""
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

        # Compute gradients for fast weights with proper error handling
        try:
            # Ensure gradients are enabled for the parameters
            slot.fast_weights.requires_grad_(True)
            slot.fast_bias.requires_grad_(True)

            weight_grad = torch.autograd.grad(loss, slot.fast_weights,
                                            retain_graph=True, create_graph=False)[0]
            bias_grad = torch.autograd.grad(loss, slot.fast_bias,
                                          retain_graph=True, create_graph=False)[0]

            # Use enhanced gradient-based update
            slot.update_with_gradients(
                weight_grad,
                bias_grad,
                self.config.meta_learning_rate,
                self.config.fast_weight_decay
            )

            weight_norm = torch.norm(weight_grad).item()
            bias_norm = torch.norm(bias_grad).item()

        except RuntimeError as e:
            # Fallback to simple update if gradient computation fails
            print(f"Gradient computation failed: {e}, using fallback update")
            weight_norm = 0.0
            bias_norm = 0.0

        # Store episodic memory
        self._store_episodic_memory(instruction, context, target)

        return {
            "slot_used": slot_idx,
            "loss": loss.item(),
            "weight_update_norm": weight_norm,
            "bias_update_norm": bias_norm,
            "plasticity_strength": slot.plasticity_strength.item()
        }

    def maml_learn_from_instruction(self,
                                   instruction: str,
                                   context: torch.Tensor,
                                   target: torch.Tensor) -> Dict[str, Any]:
        """Enhanced learning using MAML-style adaptation."""
        # Encode instruction
        instruction_tokens = self._simple_tokenize(instruction, context.size(0))
        instruction_embedding = self.instruction_encoder(instruction_tokens)

        # Find or allocate slot
        slot_idx = self._allocate_slot(instruction_embedding)
        slot = self.slots[slot_idx]

        # Perform MAML adaptation
        maml_info = self.meta_learner.maml_adaptation(
            context, instruction_embedding, target, slot
        )

        # Store episodic memory
        self._store_episodic_memory(instruction, context, target)

        return {
            "slot_used": slot_idx,
            "maml_adaptation": maml_info,
            "plasticity_strength": slot.plasticity_strength.item()
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
