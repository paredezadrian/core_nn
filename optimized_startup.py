#!/usr/bin/env python3
"""
Optimized CORE-NN Startup Script
Uses model caching for faster startup times
"""

import time
from pathlib import Path
from optimization.efficient_model import create_efficient_model

def optimized_startup(config_path="configs/laptop_optimized.yaml"):
    """Optimized startup with caching."""
    cache_dir = Path("D:\\core-nn\\cache")
    cache_index_file = cache_dir / "cache_index.json"
    
    if cache_index_file.exists():
        print("Using cached model for faster startup...")
        # Load cached model
        from implement_model_caching import ModelCache
        cache = ModelCache()
        state_dict, cache_info = cache.load_cached_model(config_path)
        
        if state_dict is not None:
            model = create_efficient_model(config_path)
            model.load_state_dict(state_dict)
            print(f"Model loaded from cache in {cache_info.get('save_time', 0):.2f}s")
            return model
    
    print("Creating new model (no cache available)...")
    return create_efficient_model(config_path)

if __name__ == "__main__":
    start_time = time.time()
    model = optimized_startup()
    total_time = time.time() - start_time
    print(f"Total startup time: {total_time:.2f}s")
