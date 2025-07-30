"""
Profiling utilities for CORE-NN.
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps
import tracemalloc


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with memory profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory tracking
        tracemalloc.start()
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            initial_gpu_memory = 0
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            final_gpu_memory = 0
        
        # Get tracemalloc stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Print memory usage
        print(f"Memory Profile for {func.__name__}:")
        print(f"  RAM: {initial_memory:.1f} MB -> {final_memory:.1f} MB (Δ{final_memory - initial_memory:+.1f} MB)")
        print(f"  GPU: {initial_gpu_memory:.1f} MB -> {final_gpu_memory:.1f} MB (Δ{final_gpu_memory - initial_gpu_memory:+.1f} MB)")
        print(f"  Peak traced: {peak / 1024 / 1024:.1f} MB")
        
        return result
    
    return wrapper


def profile_compute(func: Callable) -> Callable:
    """
    Decorator to profile compute time of a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with compute profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start timing
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print timing
        print(f"Compute Profile for {func.__name__}:")
        print(f"  Execution time: {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


@contextmanager
def ProfilerContext(profile_memory: bool = True, 
                   profile_compute: bool = True,
                   name: str = "Operation"):
    """
    Context manager for profiling operations.
    
    Args:
        profile_memory: Whether to profile memory usage
        profile_compute: Whether to profile compute time
        name: Name of the operation being profiled
    """
    # Initialize tracking
    start_time = time.time() if profile_compute else None
    
    if profile_memory:
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            initial_gpu_memory = 0
    
    try:
        yield
    finally:
        # Compute profiling
        if profile_compute:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"⏱️  {name} execution time: {execution_time:.4f} seconds")
        
        # Memory profiling
        if profile_memory:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                final_gpu_memory = 0
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"🧠 {name} memory usage:")
            print(f"   RAM: {initial_memory:.1f} MB -> {final_memory:.1f} MB (Δ{final_memory - initial_memory:+.1f} MB)")
            print(f"   GPU: {initial_gpu_memory:.1f} MB -> {final_gpu_memory:.1f} MB (Δ{final_gpu_memory - initial_gpu_memory:+.1f} MB)")
            print(f"   Peak traced: {peak / 1024 / 1024:.1f} MB")


def get_system_stats() -> Dict[str, Any]:
    """Get current system statistics."""
    # CPU and memory info
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    
    stats = {
        "cpu_percent": cpu_percent,
        "memory_total_gb": memory_info.total / 1024**3,
        "memory_available_gb": memory_info.available / 1024**3,
        "memory_used_gb": memory_info.used / 1024**3,
        "memory_percent": memory_info.percent,
    }
    
    # GPU info if available
    if torch.cuda.is_available():
        stats.update({
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        })
        
        # Per-GPU stats
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            stats[f"gpu_{i}_name"] = props.name
            stats[f"gpu_{i}_memory_total_mb"] = props.total_memory / 1024 / 1024
    else:
        stats["gpu_available"] = False
    
    return stats


def benchmark_operation(operation: Callable, 
                       num_runs: int = 10,
                       warmup_runs: int = 3) -> Dict[str, float]:
    """
    Benchmark an operation multiple times.
    
    Args:
        operation: Function to benchmark
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with benchmark statistics
    """
    import numpy as np
    
    # Warmup runs
    for _ in range(warmup_runs):
        operation()
    
    # Benchmark runs
    times = []
    memory_usages = []
    
    for _ in range(num_runs):
        # Memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
        else:
            process = psutil.Process()
            mem_before = process.memory_info().rss
        
        # Time the operation
        start_time = time.time()
        operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        # Memory after
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
        else:
            mem_after = process.memory_info().rss
        
        times.append(end_time - start_time)
        memory_usages.append((mem_after - mem_before) / 1024 / 1024)  # MB
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "mean_memory_mb": np.mean(memory_usages),
        "std_memory_mb": np.std(memory_usages),
        "num_runs": num_runs
    }
