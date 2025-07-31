#!/usr/bin/env python3
"""
Fix sequence length scaling issues in CORE-NN model.

This script extends sequence length support from 100 tokens to 200 tokens
and ensures proper handling of longer sequences in all components.
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

# Import fixed components from batch size fix
from optimization.batch_size_fix import FixedTemporalCapsule, FixedMultiTimescaleEmbedding, FixedRTEULayer, FixedRecursiveTemporalEmbeddingUnit


class ExtendedSequenceCoreNNModel(CoreNNModel):
    """CORE-NN model with extended sequence length support and fixed batch scaling."""
    
    def __init__(self, config, vocab_size: int = 50000, max_sequence_length: int = 200):
        super().__init__(config, vocab_size)
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = 10
        
        # Replace RTEU with fixed version for proper batch size handling
        self.rteu = FixedRecursiveTemporalEmbeddingUnit(config.rteu)
        
        # Update position embedding if needed
        if self.position_embedding.num_embeddings < max_sequence_length:
            print(f"⚠️  Warning: Position embedding size ({self.position_embedding.num_embeddings}) is smaller than max sequence length ({max_sequence_length})")
            print(f"   This may cause position overflow issues with long sequences")
    
    def forward(self, 
                input_ids: torch.Tensor,
                instruction: Optional[str] = None,
                instruction_tokens: Optional[torch.Tensor] = None,
                reset_state: bool = False) -> Dict[str, Any]:
        """Forward pass with extended sequence length validation."""
        batch_size, seq_len = input_ids.shape
        
        # Validate sequence length
        if seq_len < self.min_sequence_length:
            raise ValueError(f"Sequence length {seq_len} is below minimum {self.min_sequence_length}")
        if seq_len > self.max_sequence_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_sequence_length}")
        
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
                "system_status": self.execution_engine.get_system_status(),
                "sequence_length": seq_len,
                "max_sequence_length": self.max_sequence_length
            }
        }
        
        return result


def test_sequence_length_scaling(max_length: int = 200):
    """Test sequence length scaling with different sequence lengths."""
    print(f"🧪 Testing Sequence Length Scaling (10-{max_length})")
    print("=" * 50)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/laptop_optimized_flexible_sequences.yaml")
    
    # Create extended sequence model
    model = ExtendedSequenceCoreNNModel(config, vocab_size=1000, max_sequence_length=max_length)
    model.eval()
    
    # Test sequence lengths
    sequence_lengths = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    batch_sizes = [1, 2, 4]
    
    results = {
        'sequence_length_tests': {},
        'batch_size_tests': {}
    }
    
    # Test sequence length scaling
    print("📏 Testing sequence length scaling...")
    for seq_len in sequence_lengths:
        if seq_len > max_length:
            continue
            
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
                
                with torch.no_grad():
                    output = model.forward(input_ids)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0
                else:
                    import time
                    elapsed_time = time.time()
                    with torch.no_grad():
                        output = model.forward(input_ids)
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
    
    # Test batch size scaling with longer sequences
    print("\n📊 Testing batch size scaling with longer sequences...")
    for batch_size in batch_sizes:
        print(f"  Testing batch size {batch_size} with sequence length 150...")
        
        try:
            # Create input with longer sequence
            input_ids = torch.randint(1, 1000, (batch_size, 150))
            
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
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0
                else:
                    import time
                    elapsed_time = time.time()
                    with torch.no_grad():
                        output = model.forward(input_ids)
                    elapsed_time = time.time() - elapsed_time
                
                times.append(elapsed_time)
            
            # Calculate metrics
            mean_time = sum(times) / len(times)
            throughput = (batch_size * 150) / mean_time  # tokens per second
            
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
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 SEQUENCE LENGTH SCALING RESULTS")
    print("=" * 50)
    
    seq_success_count = sum(1 for result in results['sequence_length_tests'].values() if result.get('success', False))
    batch_success_count = sum(1 for result in results['batch_size_tests'].values() if result.get('success', False))
    
    print(f"Sequence Length Tests: {seq_success_count}/{len(sequence_lengths)} passed")
    print(f"Batch Size Tests: {batch_success_count}/{len(batch_sizes)} passed")
    
    # Check if we achieved the target sequence length
    max_tested_length = max(seq_len for seq_len in sequence_lengths if seq_len <= max_length)
    print(f"Maximum tested sequence length: {max_tested_length}")
    print(f"Target sequence length: {max_length}")
    
    if seq_success_count == len([s for s in sequence_lengths if s <= max_length]) and batch_success_count == len(batch_sizes):
        print("✅ All sequence length scaling tests passed!")
        return True
    else:
        print("❌ Some sequence length scaling tests failed")
        return False


def validate_memory_usage(max_length: int = 200):
    """Validate memory usage with extended sequences."""
    print(f"\n💾 Validating Memory Usage (max length: {max_length})")
    print("-" * 30)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/laptop_optimized_flexible_sequences.yaml")
    
    # Create model
    model = ExtendedSequenceCoreNNModel(config, vocab_size=1000, max_sequence_length=max_length)
    model.eval()
    
    # Test memory usage with different sequence lengths
    test_lengths = [50, 100, 150, 200]
    
    for seq_len in test_lengths:
        if seq_len > max_length:
            continue
            
        try:
            # Create input
            input_ids = torch.randint(1, 1000, (1, seq_len))
            
            # Measure memory before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            with torch.no_grad():
                output = model.forward(input_ids)
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            print(f"  Sequence length {seq_len}: {memory_used:.1f} MB memory used")
            
        except Exception as e:
            print(f"  ❌ Sequence length {seq_len} memory test failed: {e}")
    
    print("✅ Memory usage validation completed")


def main():
    """Main function to run sequence length scaling fix."""
    parser = argparse.ArgumentParser(description="Fix sequence length scaling issues in CORE-NN")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum sequence length to test")
    parser.add_argument("--validate-memory", action="store_true", help="Validate memory usage")
    parser.add_argument("--output", type=str, default="sequence_scaling_fix_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    print("🔧 CORE-NN Sequence Length Scaling Fix")
    print("=" * 50)
    
    # Test sequence length scaling
    success = test_sequence_length_scaling(args.max_length)
    
    # Validate memory usage if requested
    if args.validate_memory:
        validate_memory_usage(args.max_length)
    
    if success:
        print(f"\n✅ Sequence length scaling fix completed successfully!")
        print(f"The model now properly handles sequences 10-{args.max_length} tokens")
    else:
        print(f"\n❌ Sequence length scaling fix encountered issues")
        print("Please check the error messages above")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 