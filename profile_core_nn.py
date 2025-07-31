#!/usr/bin/env python3
"""
CORE-NN Performance Profiling Script
Tests performance on laptop hardware (Intel i5-11320H, 16GB RAM)
"""

import time
import json
import psutil
import torch
from core_nn.utils.profiling import get_system_stats, benchmark_operation
from core_nn import CoreNNModel
from core_nn.config.manager import ConfigManager

def profile_core_nn_components():
    """Profile individual CORE-NN components"""
    print("🔍 Profiling CORE-NN Components...")
    
    # Get system stats
    system_stats = get_system_stats()
    print(f"System Stats: {json.dumps(system_stats, indent=2)}")
    
    # Test basic tensor operations
    def test_tensor_ops():
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        return torch.mm(x, y)
    
    tensor_result = benchmark_operation(test_tensor_ops, num_runs=5)
    print(f"Tensor Operations: {tensor_result['mean_time']:.4f}s avg")
    
    # Test model initialization
    def test_model_init():
        config_manager = ConfigManager()
        config = config_manager.load_config('configs/default.yaml')
        model = CoreNNModel(config)
        return model
    
    try:
        model_result = benchmark_operation(test_model_init, num_runs=3)
        print(f"Model Initialization: {model_result['mean_time']:.4f}s avg")
    except Exception as e:
        print(f"Model initialization failed: {e}")
    
    # Test inference if model loads successfully
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config('configs/default.yaml')
        model = CoreNNModel(config)
        
        def test_inference():
            input_ids = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                return model.generate(input_ids, max_new_tokens=5)
        
        inference_result = benchmark_operation(test_inference, num_runs=3)
        print(f"Inference: {inference_result['mean_time']:.4f}s avg")
        
    except Exception as e:
        print(f"Inference test failed: {e}")
    
    return {
        'system_stats': system_stats,
        'tensor_ops': tensor_result,
        'model_init': model_result if 'model_result' in locals() else None,
        'inference': inference_result if 'inference_result' in locals() else None
    }

def profile_memory_usage():
    """Profile memory usage patterns"""
    print("\n🧠 Profiling Memory Usage...")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test memory usage with different tensor sizes
    memory_tests = []
    
    for size in [100, 500, 1000, 2000]:
        def test_memory(size=size):
            tensors = [torch.randn(size, size) for _ in range(5)]
            return sum(t.numel() for t in tensors)
        
        before_memory = process.memory_info().rss / 1024 / 1024
        result = test_memory(size)
        after_memory = process.memory_info().rss / 1024 / 1024
        
        memory_tests.append({
            'size': size,
            'memory_before': before_memory,
            'memory_after': after_memory,
            'memory_delta': after_memory - before_memory,
            'tensor_elements': result
        })
        
        print(f"Size {size}x{size}: {before_memory:.1f}MB -> {after_memory:.1f}MB (Δ{after_memory - before_memory:+.1f}MB)")
    
    return memory_tests

def profile_cpu_utilization():
    """Profile CPU utilization"""
    print("\n⚡ Profiling CPU Utilization...")
    
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Frequency: {cpu_freq.current:.1f} MHz")
    
    # Test CPU-intensive operation
    def cpu_intensive():
        result = 0
        for i in range(1000000):
            result += i * i
        return result
    
    cpu_result = benchmark_operation(cpu_intensive, num_runs=3)
    print(f"CPU-intensive operation: {cpu_result['mean_time']:.4f}s avg")
    
    return {
        'cpu_percent': cpu_percent,
        'cpu_count': cpu_count,
        'cpu_freq': cpu_freq.current if cpu_freq else None,
        'cpu_benchmark': cpu_result
    }

def main():
    """Main profiling function"""
    print("🚀 CORE-NN Performance Profiling")
    print("=" * 50)
    
    # Run all profiling tests
    component_results = profile_core_nn_components()
    memory_results = profile_memory_usage()
    cpu_results = profile_cpu_utilization()
    
    # Compile results
    all_results = {
        'timestamp': time.time(),
        'system_info': {
            'cpu': 'Intel i5-11320H',
            'ram': '16GB',
            'gpu': 'Intel Iris Xe Graphics (CPU-only)',
            'os': 'Windows 10'
        },
        'component_profiling': component_results,
        'memory_profiling': memory_results,
        'cpu_profiling': cpu_results
    }
    
    # Save results
    with open('profile_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Profiling complete! Results saved to profile_results.json")
    
    # Print summary
    print("\n📊 Performance Summary:")
    print(f"  • Tensor operations: {component_results['tensor_ops']['mean_time']:.4f}s")
    if component_results['model_init']:
        print(f"  • Model initialization: {component_results['model_init']['mean_time']:.4f}s")
    if component_results['inference']:
        print(f"  • Inference: {component_results['inference']['mean_time']:.4f}s")
    print(f"  • CPU utilization: {cpu_results['cpu_percent']}%")
    print(f"  • Available memory: {component_results['system_stats']['memory_available_gb']:.1f}GB")

if __name__ == "__main__":
    main() 