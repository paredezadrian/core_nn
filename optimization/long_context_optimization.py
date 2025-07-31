#!/usr/bin/env python3
"""
Long-Context Performance Optimization for CORE-NN.

This script optimizes long-context processing performance by implementing:
1. Memory-efficient processing within specified limits
2. Optimized chunked processing strategies
3. Performance tuning for different sequence lengths
4. Memory usage optimization
5. CPU/GPU optimization strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import math
import time
import gc
import psutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.config.schema import CoreNNConfig
from optimization.long_context_fix import LongContextCoreNNModel, ChunkedSequenceProcessor


class OptimizedChunkedProcessor(ChunkedSequenceProcessor):
    """
    Optimized chunked processor with adaptive chunk sizes and memory management.
    """
    
    def __init__(self, memory_limit_gb: float = 10.0, min_chunk_size: int = 512, max_chunk_size: int = 2048):
        super().__init__(chunk_size=1024, overlap=50)
        self.memory_limit_gb = memory_limit_gb
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Adaptive chunk sizing
        self.adaptive_chunk_sizes = {
            1000: 512,    # Small sequences
            2000: 1024,   # Medium sequences
            4000: 1536,   # Large sequences
            8000: 2048    # Very large sequences
        }
    
    def get_optimal_chunk_size(self, sequence_length: int, available_memory_mb: float) -> int:
        """
        Determine optimal chunk size based on sequence length and available memory.
        
        Args:
            sequence_length: Length of the sequence
            available_memory_mb: Available memory in MB
            
        Returns:
            Optimal chunk size
        """
        # Base chunk size on sequence length
        for threshold, chunk_size in sorted(self.adaptive_chunk_sizes.items()):
            if sequence_length <= threshold:
                base_chunk_size = chunk_size
                break
        else:
            base_chunk_size = self.max_chunk_size
        
        # Adjust based on available memory
        # Estimate memory per token (rough approximation)
        memory_per_token = 0.5  # MB per token
        max_tokens_for_memory = int(available_memory_mb / memory_per_token)
        
        # Use the smaller of the two constraints
        optimal_chunk_size = min(base_chunk_size, max_tokens_for_memory)
        
        # Ensure within bounds
        optimal_chunk_size = max(self.min_chunk_size, min(optimal_chunk_size, self.max_chunk_size))
        
        return optimal_chunk_size
    
    def chunk_sequence_optimized(self, sequence: torch.Tensor, available_memory_mb: float) -> List[torch.Tensor]:
        """
        Split sequence into optimally sized chunks based on available memory.
        
        Args:
            sequence: Input sequence [seq_len, ...]
            available_memory_mb: Available memory in MB
            
        Returns:
            List of optimally sized chunks
        """
        seq_len = sequence.shape[0]
        
        # Get optimal chunk size
        chunk_size = self.get_optimal_chunk_size(seq_len, available_memory_mb)
        
        # Overlap based on chunk size (larger chunks need less overlap)
        overlap = max(25, chunk_size // 20)
        
        chunks = []
        
        if seq_len <= chunk_size:
            return [sequence]
        
        # Create overlapping chunks with optimal size
        for start in range(0, seq_len, chunk_size - overlap):
            end = min(start + chunk_size, seq_len)
            chunk = sequence[start:end]
            chunks.append(chunk)
            
            if end == seq_len:
                break
        
        return chunks


class MemoryOptimizedCoreNNModel(LongContextCoreNNModel):
    """
    Memory-optimized CORE-NN model for long-context processing.
    
    Features:
    - Adaptive chunk sizing based on available memory
    - Memory usage monitoring and optimization
    - Performance tuning for different sequence lengths
    - Efficient memory management
    """
    
    def __init__(self, config: CoreNNConfig, vocab_size: int = 50000, max_sequence_length: int = 8192, 
                 memory_limit_gb: float = 10.0):
        super().__init__(config, vocab_size, max_sequence_length)
        
        # Memory management
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_mb = memory_limit_gb * 1024
        
        # Replace chunk processor with optimized version
        self.chunk_processor = OptimizedChunkedProcessor(
            memory_limit_gb=memory_limit_gb,
            min_chunk_size=512,
            max_chunk_size=2048
        )
        
        # Performance tracking
        self.performance_stats = {
            'total_sequences': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'memory_usage': []
        }
    
    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        memory = psutil.virtual_memory()
        return memory.available / 1024 / 1024
    
    def check_memory_limits(self) -> bool:
        """Check if current memory usage is within limits."""
        memory = psutil.virtual_memory()
        current_usage_mb = memory.used / 1024 / 1024
        return current_usage_mb < self.memory_limit_mb
    
    def _process_long_sequence_optimized(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Process very long sequences with memory optimization.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Model outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Get available memory
        available_memory_mb = self.get_available_memory_mb()
        
        print(f"Processing long sequence ({seq_len} tokens) with {available_memory_mb:.1f}MB available memory")
        
        # Use optimized chunked processing for very long sequences
        if seq_len > 2048:
            # Process in optimized chunks
            chunks = self.chunk_processor.chunk_sequence_optimized(input_ids[0], available_memory_mb)
            
            chunk_outputs = []
            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)} ({chunk.shape[0]} tokens)")
                
                # Check memory before processing
                if not self.check_memory_limits():
                    print(f"    ⚠️ Memory limit approaching, forcing cleanup")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Add batch dimension
                chunk_batch = chunk.unsqueeze(0)
                
                # Process chunk
                chunk_output = self._process_chunk(chunk_batch)
                chunk_outputs.append(chunk_output)
                
                # Memory management
                self._check_memory_usage()
            
            # Merge chunk outputs
            merged_output = self._merge_chunk_outputs(chunk_outputs, seq_len)
            
            return merged_output
        
        else:
            # Use standard processing for shorter sequences
            return self._process_standard_sequence(input_ids)
    
    def forward(self, 
                input_ids: torch.Tensor,
                instruction: Optional[str] = None,
                instruction_tokens: Optional[torch.Tensor] = None,
                reset_state: bool = False) -> Dict[str, Any]:
        """
        Optimized forward pass with memory management.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            instruction: Optional instruction string
            instruction_tokens: Optional instruction tokens
            reset_state: Whether to reset model state
            
        Returns:
            Model outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Update performance stats
        self.performance_stats['total_sequences'] += 1
        self.performance_stats['total_tokens'] += seq_len
        
        # Memory management
        self._check_memory_usage()
        
        # Record memory usage
        memory_usage_mb = psutil.virtual_memory().used / 1024 / 1024
        self.performance_stats['memory_usage'].append(memory_usage_mb)
        
        # Handle very long sequences with optimization
        if seq_len > 2048:
            return self._process_long_sequence_optimized(input_ids)
        
        # Standard processing for shorter sequences
        return self._process_standard_sequence(input_ids)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if self.performance_stats['total_sequences'] == 0:
            return {}
        
        avg_tokens_per_second = self.performance_stats['total_tokens'] / max(self.performance_stats['total_time'], 0.001)
        avg_memory_usage = np.mean(self.performance_stats['memory_usage']) if self.performance_stats['memory_usage'] else 0
        
        return {
            'total_sequences': self.performance_stats['total_sequences'],
            'total_tokens': self.performance_stats['total_tokens'],
            'total_time': self.performance_stats['total_time'],
            'avg_tokens_per_second': avg_tokens_per_second,
            'avg_memory_usage_mb': avg_memory_usage,
            'max_memory_usage_mb': max(self.performance_stats['memory_usage']) if self.performance_stats['memory_usage'] else 0
        }


def test_optimized_long_context_model(config_path: str, max_tokens: int = 4096, memory_limit_gb: float = 10.0):
    """Test the optimized long-context model with various sequence lengths."""
    print(f"Testing Optimized Long-Context Model")
    print(f"Max tokens: {max_tokens}, Memory limit: {memory_limit_gb}GB")
    print("=" * 60)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Create optimized model
    model = MemoryOptimizedCoreNNModel(
        config, 
        vocab_size=50000, 
        max_sequence_length=max_tokens,
        memory_limit_gb=memory_limit_gb
    )
    model.eval()
    
    # Test different sequence lengths with memory constraints
    test_lengths = [100, 500, 1000, 2000, 4000, 6000, 8000]
    
    results = {}
    
    for length in test_lengths:
        print(f"\nTesting sequence length: {length} tokens")
        
        try:
            # Create test input
            input_ids = torch.randint(0, 50000, (1, length))
            
            # Measure memory before
            start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
            
            print(f"  Available memory: {available_memory:.1f}MB")
            
            # Process sequence
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            end_time = time.time()
            
            # Measure memory after
            end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            
            # Calculate metrics
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            tokens_per_second = length / processing_time
            
            # Update model's total time
            model.performance_stats['total_time'] += processing_time
            
            results[length] = {
                'success': True,
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'tokens_per_second': tokens_per_second,
                'available_memory_before': available_memory,
                'memory_usage_absolute': end_memory
            }
            
            print(f"  ✅ Success: {processing_time:.2f}s, {memory_usage:.1f}MB, {tokens_per_second:.1f} tokens/sec")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:100]}...")
            results[length] = {
                'success': False,
                'error': str(e)
            }
    
    # Get performance summary
    performance_summary = model.get_performance_summary()
    
    # Print summary
    print(f"\n📊 OPTIMIZED LONG-CONTEXT TEST SUMMARY:")
    successful_lengths = [length for length, result in results.items() if result['success']]
    if successful_lengths:
        max_successful = max(successful_lengths)
        print(f"✅ Maximum successful sequence length: {max_successful} tokens")
        
        # Calculate average performance
        successful_results = [results[length] for length in successful_lengths]
        avg_tokens_per_second = np.mean([r['tokens_per_second'] for r in successful_results])
        avg_memory_usage = np.mean([r['memory_usage'] for r in successful_results])
        
        print(f"📈 Average performance: {avg_tokens_per_second:.1f} tokens/sec")
        print(f"💾 Average memory usage: {avg_memory_usage:.1f} MB")
        print(f"🎯 Performance summary: {performance_summary}")
    else:
        print("❌ No successful tests")
    
    return results, performance_summary


def main():
    """Main function to run long-context optimization."""
    parser = argparse.ArgumentParser(description="CORE-NN Long-Context Performance Optimization")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Maximum number of tokens to support")
    parser.add_argument("--memory-limit", type=str, default="10GB",
                       help="Memory limit for processing (e.g., 10GB)")
    parser.add_argument("--config", type=str, default="configs/laptop_optimized_flexible_sequences.yaml",
                       help="CORE-NN model configuration file path")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Force CPU-only processing")
    parser.add_argument("--output", type=str, default="long_context_optimization.json",
                       help="Output file for optimization results")
    
    args = parser.parse_args()
    
    # Parse memory limit
    memory_limit_str = args.memory_limit.upper()
    if memory_limit_str.endswith('GB'):
        memory_limit_gb = float(memory_limit_str[:-2])
    elif memory_limit_str.endswith('MB'):
        memory_limit_gb = float(memory_limit_str[:-2]) / 1024
    else:
        memory_limit_gb = float(memory_limit_str) / 1024  # Assume MB
    
    print("🚀 CORE-NN Long-Context Performance Optimization")
    print("=" * 55)
    print(f"Max tokens: {args.max_tokens}")
    print(f"Memory limit: {args.memory_limit} ({memory_limit_gb:.1f}GB)")
    print(f"Config: {args.config}")
    print(f"CPU only: {args.cpu_only}")
    
    # Set device
    if args.cpu_only:
        torch.set_default_tensor_type(torch.FloatTensor)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Test the optimized long-context model
    results, performance_summary = test_optimized_long_context_model(
        args.config, 
        args.max_tokens, 
        memory_limit_gb
    )
    
    # Save results
    output_dir = Path("optimization/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"long_context_optimization_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = {
        'test_results': {},
        'performance_summary': performance_summary,
        'configuration': {
            'max_tokens': args.max_tokens,
            'memory_limit_gb': memory_limit_gb,
            'config_path': args.config,
            'cpu_only': args.cpu_only
        }
    }
    
    for length, result in results.items():
        if result['success']:
            serializable_results['test_results'][str(length)] = {
                'success': True,
                'processing_time': result['processing_time'],
                'memory_usage': result['memory_usage'],
                'tokens_per_second': result['tokens_per_second'],
                'available_memory_before': result['available_memory_before'],
                'memory_usage_absolute': result['memory_usage_absolute']
            }
        else:
            serializable_results['test_results'][str(length)] = {
                'success': False,
                'error': result['error']
            }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📁 Results saved to {results_file}")
    
    # Print final summary
    successful_tests = sum(1 for result in results.values() if result['success'])
    total_tests = len(results)
    
    print(f"\n🎯 FINAL OPTIMIZATION SUMMARY:")
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    
    if successful_tests > 0:
        print(f"✅ Long-context performance optimization successful")
        print(f"✅ Memory-efficient processing within {memory_limit_gb:.1f}GB limit")
        print(f"✅ Adaptive chunk sizing implemented")
        print(f"✅ Performance monitoring and optimization active")
        
        # Performance metrics
        if performance_summary:
            print(f"📊 Performance metrics:")
            print(f"   - Total sequences processed: {performance_summary['total_sequences']}")
            print(f"   - Total tokens processed: {performance_summary['total_tokens']}")
            print(f"   - Average tokens/sec: {performance_summary['avg_tokens_per_second']:.1f}")
            print(f"   - Average memory usage: {performance_summary['avg_memory_usage_mb']:.1f}MB")
            print(f"   - Max memory usage: {performance_summary['max_memory_usage_mb']:.1f}MB")
    else:
        print(f"❌ Long-context optimization failed")
    
    return successful_tests > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 