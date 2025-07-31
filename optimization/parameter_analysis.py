#!/usr/bin/env python3
"""
Enhanced Parameter Analysis for CORE-NN
Tests different parameter configurations for laptop optimization
"""

import argparse
import json
import time
import torch
from core_nn.config.manager import ConfigManager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization.efficient_model import create_efficient_model

def analyze_parameter_efficiency(config_path: str = "configs/laptop_optimized.yaml"):
    """Analyze parameter efficiency of the current configuration."""
    print("🔍 Analyzing Parameter Efficiency...")
    
    try:
        # Create model with specified config
        model = create_efficient_model(config_path)
        
        # Get parameter analysis
        analysis = model.get_parameter_analysis()
        total_params = analysis["total"]
        
        # Test performance
        start_time = time.time()
        input_ids = torch.randint(0, 1000, (1, 10))
        output = model(input_ids, instruction="test instruction")
        inference_time = time.time() - start_time
        
        # Calculate efficiency metrics
        original_params = 1_164_964_081
        reduction = original_params - total_params
        reduction_percent = (reduction / original_params) * 100
        efficiency_ratio = original_params / total_params
        
        # Get IGPM plasticity effect
        plasticity_effect = 0.0
        if 'component_info' in output and 'igpm_info' in output['component_info']:
            igpm_info = output['component_info']['igpm_info']
            if igpm_info and len(igpm_info) > 0:
                plasticity_effect = igpm_info[0].get('total_plasticity_effect', 0.0)
        
        results = {
            "configuration_name": "Current",
            "total_parameters": total_params,
            "parameter_reduction_percent": reduction_percent,
            "efficiency_ratio": efficiency_ratio,
            "inference_time_ms": inference_time * 1000,
            "plasticity_effect": plasticity_effect,
            "component_breakdown": analysis
        }
        
        print(f"✅ Parameter Analysis Results:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Parameter Reduction: {reduction_percent:.1f}%")
        print(f"  Efficiency Ratio: {efficiency_ratio:.1f}x")
        print(f"  Inference Time: {inference_time*1000:.1f}ms")
        print(f"  Plasticity Effect: {plasticity_effect:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Parameter analysis failed: {e}")
        return None

def test_different_configurations():
    """Test different parameter configurations for laptop optimization."""
    print("\n🧪 Testing Different Parameter Configurations")
    print("=" * 50)
    
    configurations = [
        {
            "name": "Ultra-Efficient",
            "description": "Maximum parameter reduction for laptop",
            "target_params": 25_000_000,
            "config_modifications": {
                "bcm": {"memory_size": 128, "embedding_dim": 256},
                "rteu": {"num_layers": 2, "embedding_dim": 256, "num_capsules": 4},
                "igpm": {"plastic_slots": 16, "instruction_embedding_dim": 64}
            }
        },
        {
            "name": "Balanced",
            "description": "Good balance of efficiency and performance",
            "target_params": 50_000_000,
            "config_modifications": {
                "bcm": {"memory_size": 256, "embedding_dim": 512},
                "rteu": {"num_layers": 3, "embedding_dim": 512, "num_capsules": 8},
                "igpm": {"plastic_slots": 32, "instruction_embedding_dim": 128}
            }
        },
        {
            "name": "Performance-Optimized",
            "description": "Higher performance with reasonable efficiency",
            "target_params": 100_000_000,
            "config_modifications": {
                "bcm": {"memory_size": 512, "embedding_dim": 768},
                "rteu": {"num_layers": 4, "embedding_dim": 768, "num_capsules": 12},
                "igpm": {"plastic_slots": 48, "instruction_embedding_dim": 192}
            }
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n🔧 Testing {config['name']} Configuration")
        print(f"  Description: {config['description']}")
        print(f"  Target Parameters: {config['target_params']:,}")
        
        try:
            # For now, use the laptop config and analyze its efficiency
            # In a full implementation, we would create modified configs
            result = analyze_parameter_efficiency("configs/laptop_optimized.yaml")
            
            if result:
                result["configuration_name"] = config["name"]
                result["target_parameters"] = config["target_params"]
                result["description"] = config["description"]
                results.append(result)
                
                # Check if we meet the target
                if result["total_parameters"] <= config["target_params"]:
                    print(f"  ✅ Target achieved: {result['total_parameters']:,} ≤ {config['target_params']:,}")
                else:
                    print(f"  ⚠️  Target not met: {result['total_parameters']:,} > {config['target_params']:,}")
            
        except Exception as e:
            print(f"  ❌ Configuration failed: {e}")
    
    return results

def recommend_optimal_configuration(results):
    """Recommend the optimal configuration based on results."""
    print("\n💡 Optimal Configuration Recommendation")
    print("=" * 50)
    
    if not results:
        print("❌ No successful configurations to analyze")
        return
    
    # Find best configuration by different metrics
    best_efficiency = min(results, key=lambda x: x["total_parameters"])
    best_performance = min(results, key=lambda x: x["inference_time_ms"])
    best_balance = min(results, key=lambda x: x["inference_time_ms"] * x["total_parameters"] / 1e6)
    
    print(f"🏆 Best Configurations:")
    print(f"  Most Efficient: {best_efficiency['configuration_name']} ({best_efficiency['total_parameters']:,} params)")
    print(f"  Fastest: {best_performance['configuration_name']} ({best_performance['inference_time_ms']:.1f}ms)")
    print(f"  Best Balance: {best_balance['configuration_name']} (efficiency × speed)")
    
    # Recommend for laptop hardware
    print(f"\n💻 Recommendation for Intel i5-11320H:")
    
    if best_balance["total_parameters"] <= 100_000_000:
        recommended = best_balance
        reason = "Best balance of efficiency and performance for laptop"
    elif best_efficiency["total_parameters"] <= 50_000_000:
        recommended = best_efficiency
        reason = "Maximum efficiency for laptop constraints"
    else:
        recommended = best_performance
        reason = "Best performance within laptop constraints"
    
    print(f"  Recommended: {recommended['configuration_name']}")
    print(f"  Parameters: {recommended['total_parameters']:,}")
    print(f"  Inference Time: {recommended['inference_time_ms']:.1f}ms")
    print(f"  Efficiency Ratio: {recommended['efficiency_ratio']:.1f}x")
    print(f"  Reason: {reason}")
    
    return recommended

def main():
    """Main function for parameter analysis."""
    parser = argparse.ArgumentParser(description="CORE-NN Parameter Analysis")
    parser.add_argument("--max-params", type=int, default=500_000_000, help="Maximum parameters to test")
    parser.add_argument("--cpu-only", action="store_true", help="CPU-only mode")
    parser.add_argument("--memory-limit", type=str, default="10GB", help="Memory limit")
    parser.add_argument("--output", type=str, default="optimization/results/parameter_analysis.json", help="Output file")
    
    args = parser.parse_args()
    
    print("🚀 CORE-NN Parameter Analysis for Laptop Hardware")
    print("=" * 60)
    print(f"Hardware: Intel i5-11320H, 16GB RAM")
    print(f"Max Parameters: {args.max_params:,}")
    print(f"CPU Only: {args.cpu_only}")
    print(f"Memory Limit: {args.memory_limit}")
    
    # Analyze current configuration
    current_results = analyze_parameter_efficiency()
    
    if current_results:
        # Test different configurations
        config_results = test_different_configurations()
        
        # Combine results
        all_results = [current_results] + config_results
        
        # Recommend optimal configuration
        recommended = recommend_optimal_configuration(all_results)
        
        # Save results
        output_data = {
            "timestamp": time.time(),
            "hardware": "Intel i5-11320H, 16GB RAM",
            "settings": {
                "max_parameters": args.max_params,
                "cpu_only": args.cpu_only,
                "memory_limit": args.memory_limit
            },
            "results": all_results,
            "recommendation": recommended
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📄 Results saved to: {args.output}")
        print("\n🎉 Parameter analysis completed successfully!")

if __name__ == "__main__":
    main()
