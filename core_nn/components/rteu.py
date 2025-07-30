"""
Recursive Temporal Embedding Unit (RTEU) - Attention replacement with temporal routing.

The RTEU combines elements of RNNs, capsule networks, and latent diffusion to create
multi-timescale embeddings with temporal routing-by-agreement mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import math

from ..config.schema import RTEUConfig


class TemporalCapsule(nn.Module):
    """Individual temporal capsule with specific timescale."""
    
    def __init__(self, input_dim: int, capsule_dim: int, timescale: int):
        super().__init__()
        self.input_dim = input_dim
        self.capsule_dim = capsule_dim
        self.timescale = timescale
        
        # Transformation matrices
        self.W_transform = nn.Linear(input_dim, capsule_dim)
        self.W_temporal = nn.Linear(capsule_dim, capsule_dim)
        
        # Temporal state
        self.register_buffer('temporal_state', torch.zeros(1, capsule_dim))
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))
        
        # Activation function
        self.activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through temporal capsule."""
        batch_size = x.size(0)
        
        # Transform input
        transformed = self.W_transform(x)
        
        # Update temporal state based on timescale
        if self.step_counter % self.timescale == 0:
            # Update state for this timescale
            if self.temporal_state.size(0) != batch_size:
                self.temporal_state = self.temporal_state.expand(batch_size, -1).contiguous()
            
            # Temporal integration
            new_state = self.W_temporal(self.temporal_state) + transformed
            self.temporal_state = self.activation(new_state)
        
        self.step_counter += 1
        
        # Return current temporal state
        if self.temporal_state.size(0) != batch_size:
            return self.temporal_state.expand(batch_size, -1)
        return self.temporal_state
    
    def reset_state(self):
        """Reset temporal state."""
        self.temporal_state.zero_()
        self.step_counter.zero_()


class RoutingByAgreement(nn.Module):
    """Temporal routing-by-agreement mechanism."""
    
    def __init__(self, num_capsules: int, capsule_dim: int, num_iterations: int = 3):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_iterations = num_iterations
        
        # Routing weights
        self.W_routing = nn.Parameter(torch.randn(num_capsules, capsule_dim, capsule_dim))
        
    def forward(self, capsule_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply routing-by-agreement to capsule outputs.
        
        Args:
            capsule_outputs: [batch_size, num_capsules, capsule_dim]
            
        Returns:
            routed_output: [batch_size, capsule_dim]
        """
        batch_size, num_capsules, capsule_dim = capsule_outputs.shape
        
        # Initialize routing coefficients
        b = torch.zeros(batch_size, num_capsules, 1, device=capsule_outputs.device)
        
        for iteration in range(self.num_iterations):
            # Softmax to get routing weights
            c = F.softmax(b, dim=1)  # [batch_size, num_capsules, 1]
            
            # Weighted sum of capsule outputs
            s = torch.sum(c * capsule_outputs, dim=1)  # [batch_size, capsule_dim]
            
            # Apply squashing function
            v = self.squash(s)  # [batch_size, capsule_dim]
            
            if iteration < self.num_iterations - 1:
                # Update routing coefficients
                # Compute agreement between capsules and output
                agreement = torch.sum(capsule_outputs * v.unsqueeze(1), dim=2, keepdim=True)
                b = b + agreement
        
        return v
    
    def squash(self, s: torch.Tensor) -> torch.Tensor:
        """Squashing function for capsule outputs."""
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        scale = s_norm**2 / (1 + s_norm**2) / (s_norm + 1e-8)
        return scale * s


class MultiTimescaleEmbedding(nn.Module):
    """Multi-timescale embedding layer with temporal capsules."""
    
    def __init__(self, config: RTEUConfig):
        super().__init__()
        self.config = config
        self.temporal_scales = config.temporal_scales
        self.num_capsules = config.num_capsules
        self.capsule_dim = config.capsule_dim
        
        # Create temporal capsules for each timescale
        self.capsules = nn.ModuleList([
            TemporalCapsule(config.embedding_dim, config.capsule_dim, scale)
            for scale in config.temporal_scales
        ])
        
        # Routing mechanism
        self.routing = RoutingByAgreement(
            len(config.temporal_scales), 
            config.capsule_dim, 
            config.routing_iterations
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.capsule_dim, config.embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multi-timescale capsules."""
        batch_size = x.size(0)
        
        # Process through each temporal capsule
        capsule_outputs = []
        for capsule in self.capsules:
            output = capsule(x)
            capsule_outputs.append(output)
        
        # Stack capsule outputs
        capsule_stack = torch.stack(capsule_outputs, dim=1)  # [batch_size, num_scales, capsule_dim]
        
        # Apply routing
        routed_output = self.routing(capsule_stack)
        
        # Project back to embedding dimension
        output = self.output_proj(routed_output)
        
        return output
    
    def reset_states(self):
        """Reset all temporal states."""
        for capsule in self.capsules:
            capsule.reset_state()


class RTEULayer(nn.Module):
    """Single RTEU layer with multi-timescale processing."""
    
    def __init__(self, config: RTEUConfig):
        super().__init__()
        self.config = config
        
        # Multi-timescale embedding
        self.temporal_embedding = MultiTimescaleEmbedding(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RTEU layer."""
        # Temporal embedding with residual connection
        temporal_out = self.temporal_embedding(x)
        x = self.norm1(x + temporal_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class RecursiveTemporalEmbeddingUnit(nn.Module):
    """
    Recursive Temporal Embedding Unit (RTEU) - Replaces self-attention.
    
    Combines RNNs, capsule networks, and latent diffusion for efficient
    temporal processing with multi-timescale embeddings and routing-by-agreement.
    """
    
    def __init__(self, config: RTEUConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        
        # Stack of RTEU layers
        self.layers = nn.ModuleList([
            RTEULayer(config) for _ in range(config.num_layers)
        ])
        
        # Input projection (if needed)
        self.input_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Global temporal state for sequence processing
        self.register_buffer('global_state', torch.zeros(1, config.embedding_dim))
        
    def forward(self, 
                x: torch.Tensor, 
                reset_state: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input through RTEU stack.
        
        Args:
            x: Input tensor [batch_size, embedding_dim]
            reset_state: Whether to reset temporal states
            
        Returns:
            output: Processed output [batch_size, embedding_dim]
            info: Processing information
        """
        if reset_state:
            self.reset_all_states()
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through layers
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            layer_outputs.append(x.detach().cpu().float().numpy().mean())  # For monitoring
        
        # Update global state
        batch_size = x.size(0)
        if self.global_state.size(0) != batch_size:
            self.global_state = self.global_state.expand(batch_size, -1).contiguous()
        
        # Integrate with global state
        self.global_state = 0.9 * self.global_state + 0.1 * x.detach()
        
        # Output projection
        output = self.output_proj(x)
        
        # Prepare info
        info = {
            "layer_activations": layer_outputs,
            "global_state_norm": torch.norm(self.global_state).item(),
            "output_norm": torch.norm(output).item()
        }
        
        return output, info
    
    def reset_all_states(self):
        """Reset all temporal states in the RTEU."""
        self.global_state.zero_()
        for layer in self.layers:
            layer.temporal_embedding.reset_states()
    
    def get_temporal_states(self) -> Dict[str, torch.Tensor]:
        """Get current temporal states for inspection."""
        states = {"global_state": self.global_state.clone()}
        
        for i, layer in enumerate(self.layers):
            for j, capsule in enumerate(layer.temporal_embedding.capsules):
                key = f"layer_{i}_capsule_{j}_scale_{capsule.timescale}"
                states[key] = capsule.temporal_state.clone()
        
        return states
