#!/usr/bin/env python3
"""
Performance benchmark suite for CORE-NN.

Benchmarks various aspects of CORE-NN performance including:
- Component-level performance
- Memory usage
- Inference speed
- Scalability
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


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.device = get_optimal_device()
        
        print(f"🚀 CORE-NN Performance Benchmark")
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
                model = CoreNNModel(config, vocab_size=1000)
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
                print(f"❌ Error testing {config_name}: {e}")
                model_results[config_name] = {'error': str(e)}
        
        self.results['full_model'] = model_results
    
    def benchmark_scalability(self):
        """Benchmark scalability with different input sizes."""
        print("\n📈 Scalability Benchmarks")
        print("-" * 30)
        
        config = self._get_edge_config()
        model = CoreNNModel(config, vocab_size=1000)
        model.to(self.device)
        
        batch_sizes = [1, 2, 4, 8]
        sequence_lengths = [10, 50, 100, 200]
        
        scalability_results = {
            'batch_size': {},
            'sequence_length': {}
        }
        
        # Batch size scaling
        print("Testing batch size scaling...")
        for batch_size in batch_sizes:
            try:
                input_ids = torch.randint(1, 1000, (batch_size, 50), device=self.device)
                
                times = []
                for _ in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        model.forward(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                scalability_results['batch_size'][batch_size] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times)
                }
                
            except Exception as e:
                print(f"❌ Error with batch size {batch_size}: {e}")
        
        # Sequence length scaling
        print("Testing sequence length scaling...")
        for seq_len in sequence_lengths:
            try:
                input_ids = torch.randint(1, 1000, (1, seq_len), device=self.device)
                
                times = []
                for _ in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        model.forward(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                scalability_results['sequence_length'][seq_len] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times)
                }
                
            except Exception as e:
                print(f"❌ Error with sequence length {seq_len}: {e}")
        
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
        config = RTEUConfig(num_layers=2, embedding_dim=256, hidden_dim=512)
        rteu = RecursiveTemporalEmbeddingUnit(config)
        rteu.to(self.device)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 256, device=self.device)
            rteu(input_data)
        
        # Benchmark forward pass
        times = []
        
        for _ in range(100):
            input_data = torch.randn(1, 256, device=self.device)
            
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
        config = IGPMConfig(plastic_slots=32, instruction_embedding_dim=128)
        igpm = InstructionGuidedPlasticityModule(config, vocab_size=1000, embedding_dim=128)
        igpm.to(self.device)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 128, device=self.device)
            igpm(input_data, instruction="test")
        
        # Benchmark forward pass
        times = []
        
        for _ in range(100):
            input_data = torch.randn(1, 128, device=self.device)
            
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
        config = MLCSConfig(compression_ratio=0.5, num_compression_levels=2, latent_dim=64)
        mlcs = MultiLevelCompressionSynthesizer(config, input_dim=128)
        mlcs.to(self.device)
        
        # Test compression
        knowledge_data = torch.randn(4, 128, device=self.device)
        
        compression_times = []
        decompression_times = []
        compression_ratios = []
        
        for _ in range(20):
            # Compression
            start_time = time.time()
            kpack = mlcs.compress_knowledge(knowledge_data, name="test")
            compression_time = time.time() - start_time
            compression_times.append(compression_time)
            
            # Decompression
            start_time = time.time()
            reconstructed = mlcs.decompress_knowledge(kpack)
            decompression_time = time.time() - start_time
            decompression_times.append(decompression_time)
            
            # Compression ratio
            ratio = kpack.original_size / kpack.compressed_size
            compression_ratios.append(ratio)
        
        return {
            'compression_time_ms': np.mean(compression_times) * 1000,
            'decompression_time_ms': np.mean(decompression_times) * 1000,
            'compression_ratio': np.mean(compression_ratios),
            'compression_std': np.std(compression_ratios)
        }
    
    def _benchmark_model_inference(self, model, config_name):
        """Benchmark model inference."""
        # Generation benchmark
        input_ids = torch.randint(1, 1000, (1, 20), device=self.device)
        
        generation_times = []
        tokens_per_second = []
        
        for _ in range(10):
            start_time = time.time()
            result = model.generate(input_ids, max_new_tokens=20, temperature=0.7)
            end_time = time.time()
            
            generation_time = end_time - start_time
            generation_times.append(generation_time)
            
            num_tokens = len(result['generated_tokens'])
            tps = num_tokens / generation_time if generation_time > 0 else 0
            tokens_per_second.append(tps)
        
        return {
            'mean_generation_time_s': np.mean(generation_times),
            'std_generation_time_s': np.std(generation_times),
            'mean_tokens_per_second': np.mean(tokens_per_second),
            'std_tokens_per_second': np.std(tokens_per_second)
        }
    
    def _benchmark_model_memory(self, model, config_name):
        """Benchmark model memory operations."""
        # Memory operation times
        remember_times = []
        recall_times = []
        
        facts = [
            "The sky is blue",
            "Water is wet",
            "Fire is hot",
            "Ice is cold",
            "Grass is green"
        ]
        
        # Remember operations
        for fact in facts:
            start_time = time.time()
            model.remember(fact)
            end_time = time.time()
            remember_times.append(end_time - start_time)
        
        # Recall operations
        queries = ["sky", "water", "fire", "ice", "grass"]
        for query in queries:
            start_time = time.time()
            model.recall(query, top_k=3)
            end_time = time.time()
            recall_times.append(end_time - start_time)
        
        # Memory statistics
        memory_stats = model.get_memory_stats()
        
        return {
            'mean_remember_time_ms': np.mean(remember_times) * 1000,
            'mean_recall_time_ms': np.mean(recall_times) * 1000,
            'memory_stats': memory_stats
        }
    
    def _get_minimal_config(self):
        """Get minimal configuration."""
        config = CoreNNConfig()
        config.bcm.memory_size = 64
        config.bcm.embedding_dim = 256
        config.rteu.num_layers = 1
        config.rteu.embedding_dim = 256
        config.rteu.hidden_dim = 512
        config.igpm.plastic_slots = 16
        config.execution_engine.memory_budget_gb = 4
        return config
    
    def _get_edge_config(self):
        """Get edge device configuration."""
        config_manager = ConfigManager()
        try:
            return config_manager.load_config("configs/edge_device.yaml")
        except:
            return self._get_minimal_config()
    
    def _get_default_config(self):
        """Get default configuration."""
        config_manager = ConfigManager()
        try:
            return config_manager.load_config("configs/default.yaml")
        except:
            return CoreNNConfig()
    
    def save_results(self):
        """Save benchmark results."""
        # Save JSON results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {results_file}")
        
        # Generate plots if matplotlib is available
        try:
            self._generate_plots()
        except ImportError:
            print("📊 Matplotlib not available, skipping plots")
    
    def _generate_plots(self):
        """Generate performance plots."""
        plt.style.use('seaborn-v0_8')
        
        # Component performance comparison
        if 'bcm' in self.results and 'rteu' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            components = []
            times = []
            
            for comp_name, comp_results in self.results.items():
                if comp_name in ['bcm', 'rteu', 'igpm', 'mlcs'] and 'mean_time_ms' in comp_results:
                    components.append(comp_name.upper())
                    times.append(comp_results['mean_time_ms'])
            
            ax.bar(components, times)
            ax.set_ylabel('Mean Time (ms)')
            ax.set_title('Component Performance Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'component_performance.png', dpi=300)
            plt.close()
        
        print(f"📊 Plots saved to {self.output_dir}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*50)
        print("📋 BENCHMARK SUMMARY")
        print("="*50)
        
        # Component summary
        if any(comp in self.results for comp in ['bcm', 'rteu', 'igpm', 'mlcs']):
            print("\n🔧 Component Performance:")
            for comp_name in ['bcm', 'rteu', 'igpm', 'mlcs']:
                if comp_name in self.results and 'mean_time_ms' in self.results[comp_name]:
                    result = self.results[comp_name]
                    print(f"  {comp_name.upper()}: {result['mean_time_ms']:.2f}ms ± {result['std_time_ms']:.2f}ms")
        
        # Model summary
        if 'full_model' in self.results:
            print("\n🧠 Full Model Performance:")
            for config_name, config_results in self.results['full_model'].items():
                if 'inference' in config_results:
                    inference = config_results['inference']
                    print(f"  {config_name}: {inference['mean_tokens_per_second']:.1f} tokens/sec")
        
        print(f"\n💾 Detailed results saved to {self.output_dir}")


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    
    try:
        # Run benchmarks
        benchmark.benchmark_components()
        benchmark.benchmark_full_model()
        benchmark.benchmark_scalability()
        
        # Save and summarize results
        benchmark.save_results()
        benchmark.print_summary()
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
