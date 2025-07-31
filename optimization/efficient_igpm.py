"""
Efficient IGPM Implementation with Massive Parameter Reduction.

This module implements a parameter-efficient version of the IGPM that maintains
functionality while reducing parameters by 90%+.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class EfficientMetaLearner(nn.Module):
    """
    Highly parameter-efficient meta-learner using low-rank decomposition
    and parameter sharing techniques.
    """
    
    def __init__(self, embedding_dim: int, instruction_dim: int, rank: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.instruction_dim = instruction_dim
        self.rank = rank  # Low-rank bottleneck dimension
        
        # Context encoder with parameter sharing
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 8)
        )
        
        # Efficient meta-network using low-rank decomposition
        context_encoded_dim = embedding_dim // 8
        input_dim = context_encoded_dim + instruction_dim
        
        # Instead of generating full weight matrix, generate low-rank factors
        self.weight_generator_u = nn.Sequential(
            nn.Linear(input_dim, rank * 2),
            nn.ReLU(),
            nn.Linear(rank * 2, embedding_dim * rank)  # U matrix factors
        )
        
        self.weight_generator_v = nn.Sequential(
            nn.Linear(input_dim, rank * 2),
            nn.ReLU(),
            nn.Linear(rank * 2, rank * embedding_dim)  # V matrix factors
        )
        
        # Bias generator (much smaller)
        self.bias_generator = nn.Sequential(
            nn.Linear(input_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        # Scaling factor for stability
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, context: torch.Tensor, instruction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate efficient weight updates using low-rank decomposition."""
        batch_size = context.size(0)
        
        # Encode context efficiently
        context_encoded = self.context_encoder(context)
        
        # Combine context and instruction
        combined = torch.cat([context_encoded, instruction], dim=-1)
        
        # Generate low-rank factors
        u_factors = self.weight_generator_u(combined)  # [batch, embedding_dim * rank]
        v_factors = self.weight_generator_v(combined)  # [batch, rank * embedding_dim]
        
        # Reshape to matrices
        u_matrix = u_factors.view(batch_size, self.embedding_dim, self.rank)
        v_matrix = v_factors.view(batch_size, self.rank, self.embedding_dim)
        
        # Compute low-rank weight update: U @ V
        weight_updates = torch.bmm(u_matrix, v_matrix) * self.scale
        
        # Generate bias updates
        bias_updates = self.bias_generator(combined)
        
        return weight_updates, bias_updates


class SharedPlasticitySlot(nn.Module):
    """
    Parameter-efficient plasticity slot with shared components.
    """
    
    def __init__(self, embedding_dim: int, shared_components: Dict[str, nn.Module]):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Use shared components instead of individual ones
        self.shared_transform = shared_components['transform']
        self.shared_norm = shared_components['norm']
        
        # Only slot-specific parameters (minimal)
        self.fast_weights = nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.fast_bias = nn.Parameter(torch.zeros(embedding_dim))
        
        # Lightweight slot-specific parameters
        self.context_type = nn.Parameter(torch.randn(1))
        self.plasticity_strength = nn.Parameter(torch.ones(1) * 0.5)
        
        # Efficient gradient tracking
        self.register_buffer('weight_gradients', torch.zeros_like(self.fast_weights))
        self.register_buffer('bias_gradients', torch.zeros_like(self.fast_bias))
        
        # Usage tracking
        self.register_buffer('usage_count', torch.zeros(1))
        self.register_buffer('last_update', torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply plasticity transformation using shared components."""
        # Use shared transformation
        transformed = self.shared_transform(x)
        
        # Apply fast weights (slot-specific)
        plastic_output = F.linear(transformed, self.fast_weights, self.fast_bias)
        
        # Apply shared normalization
        output = self.shared_norm(plastic_output)
        
        return output
    
    def update_plasticity(self, weight_update: torch.Tensor, bias_update: torch.Tensor, 
                         learning_rate: float = 0.01, momentum: float = 0.9):
        """Update plasticity with efficient gradient tracking."""
        # Update gradient tracking with momentum
        self.weight_gradients.mul_(momentum).add_(weight_update, alpha=1-momentum)
        self.bias_gradients.mul_(momentum).add_(bias_update, alpha=1-momentum)
        
        # Apply updates with adaptive learning rate
        grad_norm = torch.norm(self.weight_gradients).item()
        adaptive_lr = learning_rate * max(0.1, min(2.0, grad_norm))
        
        self.fast_weights.data.add_(self.weight_gradients, alpha=adaptive_lr)
        self.fast_bias.data.add_(self.bias_gradients, alpha=adaptive_lr)
        
        # Update plasticity strength
        self.plasticity_strength.data.mul_(0.99).add_(grad_norm * 0.01)
        self.plasticity_strength.data.clamp_(0.01, 1.0)
        
        # Update usage tracking
        self.usage_count.add_(1)
        self.last_update.fill_(0.0)


class EfficientIGPM(nn.Module):
    """
    Highly parameter-efficient Instruction-Guided Plasticity Module.
    
    Key optimizations:
    1. Low-rank meta-learner (90%+ parameter reduction)
    2. Shared components across slots
    3. Efficient instruction encoding
    4. Parameter sharing strategies
    """
    
    def __init__(self, 
                 embedding_dim: int = 1536,
                 instruction_dim: int = 512,
                 num_slots: int = 8,
                 vocab_size: int = 50000,
                 rank: int = 64):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.instruction_dim = instruction_dim
        self.num_slots = num_slots
        self.rank = rank
        
        # Efficient instruction encoder with parameter sharing
        self.instruction_encoder = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim // 4),
            nn.LSTM(embedding_dim // 4, instruction_dim // 2, 
                   batch_first=True, bidirectional=True),
        )
        self.instruction_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Efficient meta-learner with massive parameter reduction
        self.meta_learner = EfficientMetaLearner(embedding_dim, instruction_dim, rank)
        
        # Shared components across all slots (major parameter saving)
        self.shared_components = nn.ModuleDict({
            'transform': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'norm': nn.LayerNorm(embedding_dim)
        })
        
        # Create slots with shared components
        self.slots = nn.ModuleList([
            SharedPlasticitySlot(embedding_dim, self.shared_components)
            for _ in range(num_slots)
        ])
        
        # Efficient slot selection mechanism
        self.slot_selector = nn.Sequential(
            nn.Linear(instruction_dim, num_slots * 2),
            nn.ReLU(),
            nn.Linear(num_slots * 2, num_slots)
        )
        
        # Output projection (shared)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def encode_instruction(self, instruction: str) -> torch.Tensor:
        """Efficiently encode instruction to embedding."""
        # Simple character-based tokenization for efficiency
        tokens = torch.tensor([min(ord(c), 49999) for c in instruction[:50]], 
                            dtype=torch.long, device=next(self.parameters()).device)
        tokens = tokens.unsqueeze(0)  # Add batch dimension
        
        # Embed and encode
        embedded = self.instruction_encoder[0](tokens)  # Embedding
        encoded, _ = self.instruction_encoder[1](embedded)  # LSTM
        
        # Pool to fixed size
        pooled = self.instruction_pooling(encoded.transpose(1, 2)).squeeze(-1)
        
        return pooled
    
    def forward(self, 
                hidden_states: torch.Tensor,
                instruction: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with efficient plasticity application."""
        batch_size, seq_len, _ = hidden_states.shape
        
        if instruction is None:
            # No plasticity, just pass through
            return {
                "hidden_states": hidden_states,
                "plasticity_effect": torch.zeros(1, device=hidden_states.device)
            }
        
        # Encode instruction efficiently
        instruction_embedding = self.encode_instruction(instruction)
        
        # Select active slots efficiently
        slot_weights = F.softmax(self.slot_selector(instruction_embedding), dim=-1)
        top_k_slots = torch.topk(slot_weights, k=min(3, self.num_slots), dim=-1)
        
        # Apply plasticity only to top-k slots (efficiency)
        plastic_outputs = []
        total_plasticity_effect = 0.0
        
        for i, (slot_idx, weight) in enumerate(zip(top_k_slots.indices[0], top_k_slots.values[0])):
            if weight > 0.1:  # Only use slots with significant weight
                slot = self.slots[slot_idx]
                
                # Apply plasticity to mean-pooled representation (efficiency)
                pooled_hidden = hidden_states.mean(dim=1)  # [batch, embedding_dim]
                
                # Generate updates using efficient meta-learner
                weight_update, bias_update = self.meta_learner(pooled_hidden, instruction_embedding)
                
                # Update slot plasticity
                slot.update_plasticity(weight_update.squeeze(0), bias_update.squeeze(0))
                
                # Apply plasticity transformation
                plastic_output = slot(pooled_hidden)
                plastic_outputs.append(plastic_output * weight)
                
                total_plasticity_effect += weight.item() * slot.plasticity_strength.item()
        
        # Combine plastic outputs
        if plastic_outputs:
            combined_plastic = torch.stack(plastic_outputs).sum(dim=0)
            # Apply to all sequence positions (broadcast)
            plastic_hidden = hidden_states + combined_plastic.unsqueeze(1)
        else:
            plastic_hidden = hidden_states
        
        # Final output projection
        output = self.output_projection(plastic_hidden)
        
        return {
            "hidden_states": output,
            "plasticity_effect": torch.tensor(total_plasticity_effect, device=hidden_states.device),
            "active_slots": top_k_slots.indices[0].tolist(),
            "slot_weights": top_k_slots.values[0].tolist()
        }
    
    def learn_from_instruction(self, 
                             instruction: str,
                             context: torch.Tensor,
                             target: torch.Tensor) -> Dict[str, float]:
        """Efficient learning from instruction with minimal computation."""
        # Encode instruction
        instruction_embedding = self.encode_instruction(instruction)
        
        # Select best slot for learning
        slot_weights = F.softmax(self.slot_selector(instruction_embedding), dim=-1)
        best_slot_idx = torch.argmax(slot_weights).item()
        best_slot = self.slots[best_slot_idx]
        
        # Generate efficient updates
        weight_update, bias_update = self.meta_learner(context, instruction_embedding)
        
        # Apply learning to best slot
        best_slot.update_plasticity(weight_update.squeeze(0), bias_update.squeeze(0))
        
        return {
            "learning_rate": 0.01,
            "slot_used": best_slot_idx,
            "plasticity_strength": best_slot.plasticity_strength.item(),
            "update_magnitude": torch.norm(weight_update).item()
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count for analysis."""
        counts = {}
        
        # Instruction encoder
        counts["instruction_encoder"] = sum(p.numel() for p in self.instruction_encoder.parameters())
        
        # Meta-learner
        counts["meta_learner"] = sum(p.numel() for p in self.meta_learner.parameters())
        
        # Shared components
        counts["shared_components"] = sum(p.numel() for p in self.shared_components.parameters())
        
        # Slots (only slot-specific parameters)
        slot_params = 0
        for slot in self.slots:
            slot_params += slot.fast_weights.numel() + slot.fast_bias.numel()
            slot_params += slot.context_type.numel() + slot.plasticity_strength.numel()
        counts["slots"] = slot_params
        
        # Other components
        counts["slot_selector"] = sum(p.numel() for p in self.slot_selector.parameters())
        counts["output_projection"] = sum(p.numel() for p in self.output_projection.parameters())
        
        counts["total"] = sum(counts.values())
        
        return counts


def create_efficient_igpm(embedding_dim: int = 1536, 
                         instruction_dim: int = 512,
                         num_slots: int = 8,
                         rank: int = 64) -> EfficientIGPM:
    """Create an efficient IGPM with massive parameter reduction."""
    return EfficientIGPM(
        embedding_dim=embedding_dim,
        instruction_dim=instruction_dim,
        num_slots=num_slots,
        rank=rank
    )


if __name__ == "__main__":
    # Test efficient IGPM
    efficient_igpm = create_efficient_igpm()
    
    # Print parameter counts
    param_counts = efficient_igpm.get_parameter_count()
    print("Efficient IGPM Parameter Counts:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,}")
    
    print(f"\nTotal parameters: {param_counts['total']:,}")
    
    # Test forward pass
    batch_size, seq_len, embedding_dim = 1, 10, 1536
    hidden_states = torch.randn(batch_size, seq_len, embedding_dim)
    
    with torch.no_grad():
        output = efficient_igpm(hidden_states, instruction="test instruction")
        print(f"\nOutput shape: {output['hidden_states'].shape}")
        print(f"Plasticity effect: {output['plasticity_effect'].item():.4f}")
