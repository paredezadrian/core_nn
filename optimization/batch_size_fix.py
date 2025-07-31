#!/usr/bin/env python3
"""
Fix batch size scaling issues in CORE-NN model.

This script addresses tensor dimension mismatches that occur when
processing different batch sizes in the RTEU components.
"""

import torch
import torch.nn as nn
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.components.rteu import TemporalCapsule, RecursiveTemporalEmbeddingUnit, RoutingByAgreement
from core_nn.config.schema import RTEUConfig


class FixedTemporalCapsule(TemporalCapsule):
    """Fixed TemporalCapsule that properly handles batch size changes."""
    
    def __init__(self, input_dim: int, capsule_dim: int, timescale: int):
        super().__init__(input_dim, capsule_dim, timescale)
        # Remove the fixed buffer registration
        self.register_buffer('temporal_state', torch.zeros(1, capsule_dim))
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through temporal capsule with proper batch handling."""
        batch_size = x.size(0)
        
        # If input has sequence dimension, take the mean across sequence
        if len(x.shape) == 3:  # [batch_size, seq_len, embedding_dim]
            x = x.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Transform input
        transformed = self.W_transform(x)
        
        # Create temporal state with correct batch size
        if not hasattr(self, '_temporal_state') or self._temporal_state.size(0) != batch_size:
            self._temporal_state = torch.zeros(batch_size, self.capsule_dim, device=x.device)
        
        # Update temporal state based on timescale
        if self.step_counter % self.timescale == 0:
            # Temporal integration
            new_state = self.W_temporal(self._temporal_state) + transformed
            self._temporal_state = self.activation(new_state)
        
        self.step_counter += 1
        
        # Return current temporal state
        return self._temporal_state.clone()
    
    def reset_state(self):
        """Reset temporal state."""
        if hasattr(self, '_temporal_state'):
            self._temporal_state.zero_()
        self.step_counter.zero_()


class FixedMultiTimescaleEmbedding(nn.Module):
    """Fixed MultiTimescaleEmbedding that properly handles batch size changes."""
    
    def __init__(self, config: RTEUConfig):
        super().__init__()
        self.config = config
        self.temporal_scales = config.temporal_scales
        self.num_capsules = config.num_capsules
        self.capsule_dim = config.capsule_dim
        
        # Create temporal capsules for each timescale
        self.capsules = nn.ModuleList([
            FixedTemporalCapsule(config.embedding_dim, config.capsule_dim, scale)
            for scale in config.temporal_scales
        ])
        
        # Routing mechanism
        self.routing = RoutingByAgreement(config.num_capsules, config.capsule_dim)
        
        # Output projection
        self.output_proj = nn.Linear(config.capsule_dim, config.embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multi-timescale capsules with proper batch handling."""
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


class FixedRTEULayer(nn.Module):
    """Fixed RTEU layer that properly handles batch size changes."""
    
    def __init__(self, config: RTEUConfig):
        super().__init__()
        self.config = config
        
        # Multi-timescale embedding with fixed components
        self.temporal_embedding = FixedMultiTimescaleEmbedding(config)
        
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


class FixedRecursiveTemporalEmbeddingUnit(RecursiveTemporalEmbeddingUnit):
    """Fixed RTEU that properly handles batch size changes."""
    
    def __init__(self, config: RTEUConfig):
        super().__init__(config)
        # Replace layers with fixed versions
        self.layers = nn.ModuleList([
            FixedRTEULayer(config) for _ in range(config.num_layers)
        ])
        # Remove the fixed buffer registration
        self.register_buffer('global_state', torch.zeros(1, config.embedding_dim))
        
    def forward(self, 
                x: torch.Tensor, 
                reset_state: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process input through RTEU stack with proper batch handling."""
        if reset_state:
            self.reset_all_states()
        
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through layers
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            layer_outputs.append(x.detach().cpu().float().numpy().mean())  # For monitoring
        
        # Create global state with correct batch size
        if not hasattr(self, '_global_state') or self._global_state.size(0) != batch_size:
            self._global_state = torch.zeros(batch_size, self.config.embedding_dim, device=x.device)
        
        # Integrate with global state
        self._global_state = 0.9 * self._global_state + 0.1 * x.detach()
        
        # Output projection
        output = self.output_proj(x)
        
        # Prepare info
        info = {
            "layer_activations": layer_outputs,
            "global_state_norm": torch.norm(self._global_state).item(),
            "output_norm": torch.norm(output).item()
        }
        
        return output, info
    
    def reset_all_states(self):
        """Reset all temporal states in the RTEU."""
        if hasattr(self, '_global_state'):
            self._global_state.zero_()
        for layer in self.layers:
            layer.temporal_embedding.reset_states()
    
    def get_temporal_states(self) -> Dict[str, torch.Tensor]:
        """Get current temporal states for inspection."""
        states = {"global_state": self._global_state.clone() if hasattr(self, '_global_state') else None}
        
        for i, layer in enumerate(self.layers):
            for j, capsule in enumerate(layer.temporal_embedding.capsules):
                key = f"layer_{i}_capsule_{j}_scale_{capsule.timescale}"
                states[key] = capsule._temporal_state.clone() if hasattr(capsule, '_temporal_state') else None
        
        return states


class BatchSizeFixedCoreNNModel(CoreNNModel):
    """CORE-NN model with fixed batch size scaling."""
    
    def __init__(self, config, vocab_size: int = 50000):
        super().__init__(config, vocab_size)
        # Replace RTEU with fixed version
        self.rteu = FixedRecursiveTemporalEmbeddingUnit(config.rteu)
    
    def forward(self, 
                input_ids: torch.Tensor,
                instruction: Optional[str] = None,
                instruction_tokens: Optional[torch.Tensor] = None,
                reset_state: bool = False) -> Dict[str, Any]:
        """Forward pass with proper batch size handling."""
        batch_size, seq_len = input_ids.shape
        
        if reset_state:
            self.reset_states()
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings with overflow handling
        max_position = self.position_embedding.num_embeddings - 1
        
        # Check if position would overflow
        if self.current_position + seq_len > max_position:
            # Reset position counter
            self.current_position = 0
        
        positions = torch.arange(self.current_position, 
                               self.current_position + seq_len, 
                               device=input_ids.device)
        position_embeds = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine embeddings
        embeddings = self.input_norm(token_embeds + position_embeds)
        
        # Process each token in sequence
        outputs = []
        component_info = {
            "bcm_info": [],
            "rteu_info": [],
            "igpm_info": [],
            "mlcs_info": []
        }
        
        for t in range(seq_len):
            current_embed = embeddings[:, t, :]  # [batch_size, embedding_dim]
            
            # 1. Process through RTEU (temporal processing)
            rteu_output, rteu_info = self.rteu(current_embed, reset_state=(t == 0 and reset_state))
            component_info["rteu_info"].append(rteu_info)
            
            # 2. Process through BCM (memory storage and retrieval)
            bcm_output, bcm_info = self.bcm(rteu_output, query_embedding=rteu_output)
            component_info["bcm_info"].append(bcm_info)
            
            # 3. Process through IGPM (instruction-guided plasticity)
            igpm_output, igpm_info = self.igpm(
                bcm_output, 
                instruction=instruction,
                instruction_tokens=instruction_tokens
            )
            component_info["igpm_info"].append(igpm_info)
            
            # 4. Final processing and output projection
            final_output = self.output_norm(igpm_output)
            logits = self.output_projection(final_output)
            
            outputs.append(logits)
        
        # Stack outputs
        output_logits = torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]
        
        # Update position counter
        self.current_position += seq_len
        
        # Prepare comprehensive output
        result = {
            "logits": output_logits,
            "last_hidden_state": final_output,
            "component_info": component_info,
            "model_info": {
                "current_position": self.current_position,
                "session_active": self.session_active,
                "memory_stats": self.get_memory_stats(),
                "system_status": self.execution_engine.get_system_status()
            }
        }
        
        return result


def test_batch_size_scaling(max_batch: int = 8):
    """Test batch size scaling with different batch sizes."""
    print(f"🧪 Testing Batch Size Scaling (1-{max_batch})")
    print("=" * 50)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/laptop_optimized_flexible_sequences.yaml")
    
    # Create fixed model
    model = BatchSizeFixedCoreNNModel(config, vocab_size=1000)
    model.eval()
    
    batch_sizes = list(range(1, max_batch + 1))
    sequence_lengths = [10, 50, 100]
    
    results = {
        'batch_size_tests': {},
        'sequence_length_tests': {}
    }
    
    # Test batch size scaling
    print("📊 Testing batch size scaling...")
    for batch_size in batch_sizes:
        print(f"  Testing batch size {batch_size}...")
        
        try:
            # Create input with fixed sequence length
            input_ids = torch.randint(1, 1000, (batch_size, 50))
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    model.forward(input_ids)
            
            # Benchmark
            times = []
            for _ in range(5):
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                
                with torch.no_grad():
                    output = model.forward(input_ids)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                else:
                    import time
                    elapsed_time = time.time()
                    with torch.no_grad():
                        output = model.forward(input_ids)
                    elapsed_time = time.time() - elapsed_time
                
                times.append(elapsed_time)
            
            # Calculate metrics
            mean_time = sum(times) / len(times)
            throughput = batch_size / mean_time  # tokens per second
            
            results['batch_size_tests'][batch_size] = {
                'mean_time': mean_time,
                'throughput': throughput,
                'success': True
            }
            
            print(f"    ✅ Batch size {batch_size}: {mean_time:.4f}s, {throughput:.1f} tokens/sec")
            
        except Exception as e:
            print(f"    ❌ Batch size {batch_size} failed: {e}")
            results['batch_size_tests'][batch_size] = {
                'error': str(e),
                'success': False
            }
    
    # Test sequence length scaling
    print("\n📏 Testing sequence length scaling...")
    for seq_len in sequence_lengths:
        print(f"  Testing sequence length {seq_len}...")
        
        try:
            # Create input with fixed batch size
            input_ids = torch.randint(1, 1000, (2, seq_len))
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    model.forward(input_ids)
            
            # Benchmark
            times = []
            for _ in range(5):
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                
                try:
                    with torch.no_grad():
                        output = model.forward(input_ids)
                except Exception as e:
                    print(f"      Debug: Error during forward pass: {e}")
                    print(f"      Debug: Input shape: {input_ids.shape}")
                    print(f"      Debug: Current position: {model.current_position}")
                    raise e
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0
                else:
                    import time
                    elapsed_time = time.time()
                    try:
                        with torch.no_grad():
                            output = model.forward(input_ids)
                    except Exception as e:
                        print(f"      Debug: Error during forward pass: {e}")
                        print(f"      Debug: Input shape: {input_ids.shape}")
                        print(f"      Debug: Current position: {model.current_position}")
                        raise e
                    elapsed_time = time.time() - elapsed_time
                
                times.append(elapsed_time)
            
            # Calculate metrics
            mean_time = sum(times) / len(times)
            throughput = (2 * seq_len) / mean_time  # tokens per second
            
            results['sequence_length_tests'][seq_len] = {
                'mean_time': mean_time,
                'throughput': throughput,
                'success': True
            }
            
            print(f"    ✅ Sequence length {seq_len}: {mean_time:.4f}s, {throughput:.1f} tokens/sec")
            
        except Exception as e:
            print(f"    ❌ Sequence length {seq_len} failed: {e}")
            results['sequence_length_tests'][seq_len] = {
                'error': str(e),
                'success': False
            }
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 BATCH SIZE SCALING RESULTS")
    print("=" * 50)
    
    batch_success_count = sum(1 for result in results['batch_size_tests'].values() if result.get('success', False))
    seq_success_count = sum(1 for result in results['sequence_length_tests'].values() if result.get('success', False))
    
    print(f"Batch Size Tests: {batch_success_count}/{len(batch_sizes)} passed")
    print(f"Sequence Length Tests: {seq_success_count}/{len(sequence_lengths)} passed")
    
    if batch_success_count == len(batch_sizes) and seq_success_count == len(sequence_lengths):
        print("✅ All batch size scaling tests passed!")
        return True
    else:
        print("❌ Some batch size scaling tests failed")
        return False


def main():
    """Main function to run batch size scaling fix."""
    parser = argparse.ArgumentParser(description="Fix batch size scaling issues in CORE-NN")
    parser.add_argument("--max-batch", type=int, default=8, help="Maximum batch size to test")
    parser.add_argument("--output", type=str, default="batch_size_fix_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    print("🔧 CORE-NN Batch Size Scaling Fix")
    print("=" * 50)
    
    # Test batch size scaling
    success = test_batch_size_scaling(args.max_batch)
    
    if success:
        print("\n✅ Batch size scaling fix completed successfully!")
        print("The model now properly handles batch sizes 1-8")
    else:
        print("\n❌ Batch size scaling fix encountered issues")
        print("Please check the error messages above")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 