#!/usr/bin/env python3
"""
Updated performance benchmark suite for CORE-NN with fixed scalability.

This benchmark integrates all the fixes from Tasks 3.1 and 3.2 to ensure
proper batch size and sequence length scaling.
"""

import torch
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import argparse

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.components.rteu import RecursiveTemporalEmbeddingUnit
from core_nn.components.igpm import InstructionGuidedPlasticityModule
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.inference import InferenceEngine
from core_nn.config.schema import *
from core_nn.utils import ProfilerContext, get_optimal_device

# Import fixed components from batch size and sequence scaling fixes
from optimization.batch_size_fix import FixedRecursiveTemporalEmbeddingUnit
from optimization.sequence_scaling_fix import ExtendedSequenceCoreNNModel


class FixedPerformanceBenchmark:
    """Updated performance benchmark suite with fixed scalability."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.device = get_optimal_device()
        
        print(f"🚀 CORE-NN Fixed Performance Benchmark")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)
    
    def benchmark_components(self):
        """Benchmark individual components."""
        print("\n📊 Component Benchmarks")
        print("-" * 30)
        
        # BCM Benchmark
        print("Testing BCM...")
        bcm_results = self._benchmark_bcm()
        self.results['bcm'] = bcm_results
        
        # RTEU Benchmark
        print("Testing RTEU...")
        rteu_results = self._benchmark_rteu()
        self.results['rteu'] = rteu_results
        
        # IGPM Benchmark
        print("Testing IGPM...")
        igpm_results = self._benchmark_igpm()
        self.results['igpm'] = igpm_results
        
        # MLCS Benchmark
        print("Testing MLCS...")
        mlcs_results = self._benchmark_mlcs()
        self.results['mlcs'] = mlcs_results
    
    def benchmark_full_model(self):
        """Benchmark full model performance."""
        print("\n🧠 Full Model Benchmarks")
        print("-" * 30)
        
        configs = {
            'minimal': self._get_minimal_config(),
            'edge': self._get_edge_config(),
            'default': self._get_default_config()
        }
        
        model_results = {}
        
        for config_name, config in configs.items():
            print(f"Testing {config_name} configuration...")
            
            try:
                # Use fixed model with extended sequence support
                model = ExtendedSequenceCoreNNModel(config, vocab_size=1000, max_sequence_length=200)
                model.to(self.device)
                
                # Benchmark inference
                inference_results = self._benchmark_model_inference(model, config_name)
                
                # Benchmark memory operations
                memory_results = self._benchmark_model_memory(model, config_name)
                
                model_results[config_name] = {
                    'inference': inference_results,
                    'memory': memory_results
                }
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Error with {config_name} configuration: {e}")
                model_results[config_name] = {'error': str(e)}
        
        self.results['full_model'] = model_results
    
    def benchmark_scalability(self):
        """Benchmark scalability with different input sizes using fixed components."""
        print("\n📈 Scalability Benchmarks (Fixed)")
        print("-" * 30)
        
        config = self._get_edge_config()
        
        # Use fixed model with extended sequence support
        model = ExtendedSequenceCoreNNModel(config, vocab_size=1000, max_sequence_length=200)
        model.to(self.device)
        
        batch_sizes = [1, 2, 4, 8]
        sequence_lengths = [10, 50, 100, 150, 200]
        
        scalability_results = {
            'batch_size': {},
            'sequence_length': {},
            'combined_scaling': {}
        }
        
        # Batch size scaling
        print("Testing batch size scaling...")
        for batch_size in batch_sizes:
            try:
                input_ids = torch.randint(1, 1000, (batch_size, 50), device=self.device)
                
                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        model.forward(input_ids)
                
                times = []
                for _ in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        model.forward(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                mean_time = np.mean(times)
                throughput = (batch_size * 50) / mean_time  # tokens per second
                
                scalability_results['batch_size'][batch_size] = {
                    'mean_time': mean_time,
                    'std_time': np.std(times),
                    'throughput': throughput
                }
                
                print(f"  ✅ Batch size {batch_size}: {mean_time:.4f}s, {throughput:.1f} tokens/sec")
                
            except Exception as e:
                print(f"  ❌ Error with batch size {batch_size}: {e}")
                scalability_results['batch_size'][batch_size] = {'error': str(e)}
        
        # Sequence length scaling
        print("\nTesting sequence length scaling...")
        for seq_len in sequence_lengths:
            try:
                input_ids = torch.randint(1, 1000, (2, seq_len), device=self.device)
                
                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        model.forward(input_ids)
                
                times = []
                for _ in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        model.forward(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                mean_time = np.mean(times)
                throughput = (2 * seq_len) / mean_time  # tokens per second
                
                scalability_results['sequence_length'][seq_len] = {
                    'mean_time': mean_time,
                    'std_time': np.std(times),
                    'throughput': throughput
                }
                
                print(f"  ✅ Sequence length {seq_len}: {mean_time:.4f}s, {throughput:.1f} tokens/sec")
                
            except Exception as e:
                print(f"  ❌ Error with sequence length {seq_len}: {e}")
                scalability_results['sequence_length'][seq_len] = {'error': str(e)}
        
        # Combined scaling (batch size with longer sequences)
        print("\nTesting combined scaling...")
        for batch_size in [1, 2, 4]:
            for seq_len in [100, 150, 200]:
                try:
                    input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=self.device)
                    
                    # Warm up
                    with torch.no_grad():
                        for _ in range(3):
                            model.forward(input_ids)
                    
                    times = []
                    for _ in range(5):
                        start_time = time.time()
                        with torch.no_grad():
                            model.forward(input_ids)
                        end_time = time.time()
                        times.append(end_time - start_time)
                    
                    mean_time = np.mean(times)
                    throughput = (batch_size * seq_len) / mean_time  # tokens per second
                    
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    scalability_results['combined_scaling'][key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'mean_time': mean_time,
                        'std_time': np.std(times),
                        'throughput': throughput
                    }
                    
                    print(f"  ✅ Batch {batch_size}, Seq {seq_len}: {mean_time:.4f}s, {throughput:.1f} tokens/sec")
                    
                except Exception as e:
                    print(f"  ❌ Error with batch {batch_size}, seq {seq_len}: {e}")
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    scalability_results['combined_scaling'][key] = {'error': str(e)}
        
        self.results['scalability'] = scalability_results
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _benchmark_bcm(self):
        """Benchmark BCM component."""
        config = BCMConfig(memory_size=256, embedding_dim=512)
        bcm = BiologicalCoreMemory(config)
        bcm.to(self.device)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 512, device=self.device)
            bcm(input_data)
        
        # Benchmark forward pass
        times = []
        memory_usages = []
        
        for _ in range(100):
            input_data = torch.randn(1, 512, device=self.device)
            
            start_time = time.time()
            output, info = bcm(input_data)
            end_time = time.time()
            
            times.append(end_time - start_time)
            memory_usages.append(info['memory_utilization'])
        
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'mean_memory_utilization': np.mean(memory_usages),
            'throughput_ops_per_sec': 1.0 / np.mean(times)
        }
    
    def _benchmark_rteu(self):
        """Benchmark RTEU component."""
        config = RTEUConfig(
            embedding_dim=512,
            hidden_dim=1024,
            num_layers=3,
            num_capsules=8,
            capsule_dim=32,
            temporal_scales=[1, 2, 4, 8],
            activation="gelu",
            dropout=0.05,
            routing_iterations=2
        )
        
        # Use fixed RTEU component
        rteu = FixedRecursiveTemporalEmbeddingUnit(config)
        rteu.to(self.device)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 512, device=self.device)
            rteu(input_data)
        
        # Benchmark forward pass
        times = []
        
        for _ in range(100):
            input_data = torch.randn(1, 512, device=self.device)
            
            start_time = time.time()
            output, info = rteu(input_data)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'throughput_ops_per_sec': 1.0 / np.mean(times)
        }
    
    def _benchmark_igpm(self):
        """Benchmark IGPM component."""
        config = IGPMConfig(
            instruction_embedding_dim=128,
            plastic_slots=32,
            max_episodic_memories=500,
            meta_learning_rate=0.0005,
            plasticity_threshold=0.85,
            fast_weight_decay=0.995
        )
        
        igpm = InstructionGuidedPlasticityModule(config, vocab_size=1000, embedding_dim=512)
        igpm.to(self.device)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 512, device=self.device)
            igpm(input_data, instruction="test instruction")
        
        # Benchmark forward pass
        times = []
        
        for _ in range(100):
            input_data = torch.randn(1, 512, device=self.device)
            
            start_time = time.time()
            output, info = igpm(input_data, instruction="test instruction")
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'throughput_ops_per_sec': 1.0 / np.mean(times)
        }
    
    def _benchmark_mlcs(self):
        """Benchmark MLCS component."""
        config = MLCSConfig(
            latent_dim=128,
            num_compression_levels=3,
            compression_ratio=0.05,
            codebook_size=4096,
            auto_compress_threshold=0.8,
            kpack_max_size_mb=25
        )
        
        mlcs = MultiLevelCompressionSynthesizer(config, input_dim=512)
        mlcs.to(self.device)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 512, device=self.device)
            mlcs.compress_knowledge(input_data, name="test")
        
        # Benchmark compression
        times = []
        
        for _ in range(100):
            input_data = torch.randn(1, 512, device=self.device)
            
            start_time = time.time()
            kpack = mlcs.compress_knowledge(input_data, name="test")
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'throughput_ops_per_sec': 1.0 / np.mean(times)
        }
    
    def _benchmark_model_inference(self, model, config_name):
        """Benchmark model inference performance."""
        # Test with different input sizes
        test_cases = [
            (1, 50),   # Small batch, medium sequence
            (2, 100),  # Medium batch, longer sequence
            (4, 150),  # Larger batch, long sequence
        ]
        
        results = {}
        
        for batch_size, seq_len in test_cases:
            try:
                input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=self.device)
                
                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        model.forward(input_ids)
                
                # Benchmark
                times = []
                for _ in range(10):
                    start_time = time.time()
                    with torch.no_grad():
                        output = model.forward(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                mean_time = np.mean(times)
                throughput = (batch_size * seq_len) / mean_time  # tokens per second
                
                results[f"batch_{batch_size}_seq_{seq_len}"] = {
                    'mean_time': mean_time,
                    'std_time': np.std(times),
                    'throughput': throughput
                }
                
            except Exception as e:
                results[f"batch_{batch_size}_seq_{seq_len}"] = {'error': str(e)}
        
        return results
    
    def _benchmark_model_memory(self, model, config_name):
        """Benchmark model memory usage."""
        # Test memory usage with different input sizes
        test_cases = [
            (1, 50),
            (2, 100),
            (4, 150),
        ]
        
        results = {}
        
        for batch_size, seq_len in test_cases:
            try:
                input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=self.device)
                
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                with torch.no_grad():
                    output = model.forward(input_ids)
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                results[f"batch_{batch_size}_seq_{seq_len}"] = {
                    'memory_used_mb': memory_used,
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after
                }
                
            except Exception as e:
                results[f"batch_{batch_size}_seq_{seq_len}"] = {'error': str(e)}
        
        return results
    
    def _get_minimal_config(self):
        """Get minimal configuration for testing."""
        config_manager = ConfigManager()
        config = config_manager.load_config("configs/laptop_optimized_flexible_sequences.yaml")
        
        # Reduce sizes for minimal testing
        config.rteu.embedding_dim = 256
        config.rteu.hidden_dim = 512
        config.rteu.num_layers = 2
        config.bcm.memory_size = 64
        config.bcm.embedding_dim = 256
        config.igpm.instruction_embedding_dim = 64
        config.mlcs.latent_dim = 64
        
        return config
    
    def _get_edge_config(self):
        """Get edge configuration for testing."""
        config_manager = ConfigManager()
        config = config_manager.load_config("configs/laptop_optimized_flexible_sequences.yaml")
        
        # Use moderate sizes for edge testing
        config.rteu.embedding_dim = 384
        config.rteu.hidden_dim = 768
        config.rteu.num_layers = 2
        config.bcm.memory_size = 128
        config.bcm.embedding_dim = 384
        config.igpm.instruction_embedding_dim = 96
        config.mlcs.latent_dim = 96
        
        return config
    
    def _get_default_config(self):
        """Get default configuration for testing."""
        config_manager = ConfigManager()
        return config_manager.load_config("configs/laptop_optimized_flexible_sequences.yaml")
    
    def save_results(self):
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"fixed_benchmark_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        results_copy = convert_dict(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        return output_file
    
    def _generate_plots(self):
        """Generate performance plots."""
        try:
            # Scalability plots
            if 'scalability' in self.results:
                scalability = self.results['scalability']
                
                # Batch size scaling plot
                if 'batch_size' in scalability:
                    batch_sizes = list(scalability['batch_size'].keys())
                    throughputs = [scalability['batch_size'][bs].get('throughput', 0) for bs in batch_sizes]
                    
                    plt.figure(figsize=(10, 6))
                    plt.subplot(2, 2, 1)
                    plt.plot(batch_sizes, throughputs, 'bo-')
                    plt.xlabel('Batch Size')
                    plt.ylabel('Throughput (tokens/sec)')
                    plt.title('Batch Size Scaling')
                    plt.grid(True)
                
                # Sequence length scaling plot
                if 'sequence_length' in scalability:
                    seq_lengths = list(scalability['sequence_length'].keys())
                    throughputs = [scalability['sequence_length'][sl].get('throughput', 0) for sl in seq_lengths]
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(seq_lengths, throughputs, 'ro-')
                    plt.xlabel('Sequence Length')
                    plt.ylabel('Throughput (tokens/sec)')
                    plt.title('Sequence Length Scaling')
                    plt.grid(True)
                
                # Component performance plot
                if 'bcm' in self.results and 'rteu' in self.results and 'igpm' in self.results:
                    components = ['BCM', 'RTEU', 'IGPM', 'MLCS']
                    times = [
                        self.results['bcm']['mean_time_ms'],
                        self.results['rteu']['mean_time_ms'],
                        self.results['igpm']['mean_time_ms'],
                        self.results['mlcs']['mean_time_ms']
                    ]
                    
                    plt.subplot(2, 2, 3)
                    plt.bar(components, times)
                    plt.xlabel('Component')
                    plt.ylabel('Mean Time (ms)')
                    plt.title('Component Performance')
                    plt.xticks(rotation=45)
                
                # Memory usage plot
                if 'full_model' in self.results:
                    configs = list(self.results['full_model'].keys())
                    memory_usage = []
                    for config in configs:
                        if 'memory' in self.results['full_model'][config]:
                            memory = self.results['full_model'][config]['memory']
                            if 'batch_1_seq_50' in memory:
                                memory_usage.append(memory['batch_1_seq_50'].get('memory_used_mb', 0))
                            else:
                                memory_usage.append(0)
                        else:
                            memory_usage.append(0)
                    
                    plt.subplot(2, 2, 4)
                    plt.bar(configs, memory_usage)
                    plt.xlabel('Configuration')
                    plt.ylabel('Memory Usage (MB)')
                    plt.title('Memory Usage by Configuration')
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                plot_file = self.output_dir / f"fixed_benchmark_plots_{time.strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"📊 Plots saved to {plot_file}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots: {e}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 50)
        print("📋 FIXED BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Component performance
        if 'bcm' in self.results and 'rteu' in self.results and 'igpm' in self.results:
            print("\n🔧 Component Performance:")
            print(f"  BCM: {self.results['bcm']['mean_time_ms']:.2f}ms ± {self.results['bcm']['std_time_ms']:.2f}ms")
            print(f"  RTEU: {self.results['rteu']['mean_time_ms']:.2f}ms ± {self.results['rteu']['std_time_ms']:.2f}ms")
            print(f"  IGPM: {self.results['igpm']['mean_time_ms']:.2f}ms ± {self.results['igpm']['std_time_ms']:.2f}ms")
            print(f"  MLCS: {self.results['mlcs']['mean_time_ms']:.2f}ms ± {self.results['mlcs']['std_time_ms']:.2f}ms")
        
        # Scalability results
        if 'scalability' in self.results:
            scalability = self.results['scalability']
            
            print("\n📈 Scalability Results:")
            
            # Batch size scaling
            if 'batch_size' in scalability:
                batch_success = sum(1 for result in scalability['batch_size'].values() if 'error' not in result)
                print(f"  Batch Size Tests: {batch_success}/{len(scalability['batch_size'])} passed")
            
            # Sequence length scaling
            if 'sequence_length' in scalability:
                seq_success = sum(1 for result in scalability['sequence_length'].values() if 'error' not in result)
                print(f"  Sequence Length Tests: {seq_success}/{len(scalability['sequence_length'])} passed")
            
            # Combined scaling
            if 'combined_scaling' in scalability:
                combined_success = sum(1 for result in scalability['combined_scaling'].values() if 'error' not in result)
                print(f"  Combined Scaling Tests: {combined_success}/{len(scalability['combined_scaling'])} passed")
        
        # Full model results
        if 'full_model' in self.results:
            print("\n🧠 Full Model Performance:")
            for config_name, results in self.results['full_model'].items():
                if 'error' not in results:
                    print(f"  {config_name}: Configuration tested successfully")
                else:
                    print(f"  {config_name}: Error - {results['error']}")
        
        print(f"\n💾 Detailed results saved to benchmark_results")


def main():
    """Main function to run fixed performance benchmark."""
    parser = argparse.ArgumentParser(description="Run fixed performance benchmark for CORE-NN")
    parser.add_argument("--cpu-focus", action="store_true", help="Focus on CPU performance")
    parser.add_argument("--detailed-timing", action="store_true", help="Include detailed timing information")
    parser.add_argument("--output", type=str, default="fixed_benchmark.json", help="Output file for results")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    if args.cpu_focus:
        print("🔧 CPU-focused benchmarking enabled")
    
    # Create benchmark instance
    benchmark = FixedPerformanceBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_components()
    benchmark.benchmark_full_model()
    benchmark.benchmark_scalability()
    
    # Save results
    output_file = benchmark.save_results()
    
    # Generate plots
    if not args.skip_plots:
        benchmark._generate_plots()
    
    # Print summary
    benchmark.print_summary()
    
    print("\n✅ Fixed performance benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 