"""
Transformer Baseline Model for CORE-NN Comparison.

This module implements a standard transformer architecture with similar
parameter count and complexity to CORE-NN for fair performance comparison
on GLUE benchmark tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for transformer baseline."""
    vocab_size: int = 50000
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    max_sequence_length: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        
        assert config.embedding_dim % config.num_heads == 0
        
        self.query = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.config.embedding_dim
        )
        
        return self.output(attn_output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.embedding_dim, config.ff_dim)
        self.linear2 = nn.Linear(config.ff_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerBaseline(nn.Module):
    """
    Standard Transformer model for GLUE benchmark comparison.
    
    This implementation provides a fair baseline for comparing CORE-NN
    performance on instruction-following tasks.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                instruction: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            instruction: Instruction string (for compatibility with CORE-NN)
            
        Returns:
            Dictionary with logits and hidden states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Expand attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.expand(
            batch_size, self.config.num_heads, seq_len, seq_len
        )
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Output projection (for language modeling)
        logits = torch.matmul(hidden_states, self.token_embedding.weight.t())
        
        return {
            "logits": logits,
            "last_hidden_state": hidden_states,
            "hidden_states": hidden_states
        }
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for comparison."""
        return {
            "model_type": "transformer_baseline",
            "num_layers": self.config.num_layers,
            "embedding_dim": self.config.embedding_dim,
            "num_heads": self.config.num_heads,
            "ff_dim": self.config.ff_dim,
            "total_parameters": self.get_parameter_count(),
            "vocab_size": self.config.vocab_size
        }


def create_transformer_baseline(embedding_dim: int = 512, 
                              num_layers: int = 6,
                              vocab_size: int = 50000) -> TransformerBaseline:
    """
    Create transformer baseline with similar complexity to CORE-NN.
    
    Args:
        embedding_dim: Embedding dimension (should match CORE-NN)
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size
        
    Returns:
        Configured transformer baseline model
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=8,
        ff_dim=embedding_dim * 4,  # Standard transformer ratio
        max_sequence_length=512,
        dropout=0.1
    )
    
    return TransformerBaseline(config)


if __name__ == "__main__":
    # Test transformer baseline
    model = create_transformer_baseline()
    print(f"Transformer Baseline Info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size, seq_len = 1, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids)
        print(f"Output logits shape: {output['logits'].shape}")
        print(f"Hidden states shape: {output['last_hidden_state'].shape}")
