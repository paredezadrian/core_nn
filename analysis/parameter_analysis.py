#!/usr/bin/env python3
"""
Parameter Analysis for CORE-NN.

This script analyzes parameter counts by component to identify which parts
of the model are using excessive parameters and preventing the 95.4% reduction target.
"""

import torch
import torch.nn as nn
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.config.schema import CoreNNConfig


def count_parameters_by_module(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters by module name.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary mapping module names to parameter counts
    """
    param_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                param_counts[name] = param_count
    
    return param_counts


def analyze_component_parameters(model: CoreNNModel) -> Dict[str, Any]:
    """
    Analyze parameters by CORE-NN components.
    
    Args:
        model: CORE-NN model to analyze
        
    Returns:
        Dictionary with detailed parameter analysis
    """
    analysis = {
        "total_parameters": 0,
        "component_breakdown": {},
        "module_breakdown": {},
        "largest_modules": [],
        "parameter_efficiency": {}
    }
    
    # Get detailed parameter counts
    module_params = count_parameters_by_module(model)
    analysis["module_breakdown"] = module_params
    
    # Group by CORE-NN components
    component_params = defaultdict(int)
    
    for module_name, param_count in module_params.items():
        if "bcm" in module_name.lower():
            component_params["BCM"] += param_count
        elif "rteu" in module_name.lower() or "routing" in module_name.lower():
            component_params["RTEU"] += param_count
        elif "igpm" in module_name.lower() or "plastic" in module_name.lower():
            component_params["IGPM"] += param_count
        elif "mlcs" in module_name.lower() or "compression" in module_name.lower():
            component_params["MLCS"] += param_count
        elif "embedding" in module_name.lower():
            component_params["Embeddings"] += param_count
        elif "position" in module_name.lower():
            component_params["Position Embeddings"] += param_count
        elif "token" in module_name.lower():
            component_params["Token Embeddings"] += param_count
        elif "output" in module_name.lower() or "head" in module_name.lower():
            component_params["Output Layers"] += param_count
        else:
            component_params["Other"] += param_count
    
    analysis["component_breakdown"] = dict(component_params)
    analysis["total_parameters"] = sum(component_params.values())
    
    # Find largest modules
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    analysis["largest_modules"] = sorted_modules[:10]
    
    # Calculate efficiency metrics
    total_params = analysis["total_parameters"]
    target_params = 53_000_000  # 53M target
    
    analysis["parameter_efficiency"] = {
        "current_total": total_params,
        "target_total": target_params,
        "reduction_needed": total_params - target_params,
        "reduction_percentage": ((total_params - target_params) / total_params) * 100,
        "efficiency_ratio": target_params / total_params
    }
    
    return analysis


def identify_optimization_targets(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify which components need optimization to reach the 53M target.
    
    Args:
        analysis: Parameter analysis results
        
    Returns:
        List of optimization targets with recommendations
    """
    targets = []
    total_params = analysis["total_parameters"]
    target_params = 53_000_000
    reduction_needed = total_params - target_params
    
    # Sort components by parameter count
    sorted_components = sorted(
        analysis["component_breakdown"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for component_name, param_count in sorted_components:
        if param_count > 0:
            percentage = (param_count / total_params) * 100
            potential_reduction = min(param_count, reduction_needed)
            
            target = {
                "component": component_name,
                "current_parameters": param_count,
                "percentage_of_total": percentage,
                "potential_reduction": potential_reduction,
                "reduction_percentage": (potential_reduction / param_count) * 100,
                "recommendations": []
            }
            
            # Generate specific recommendations
            if component_name == "RTEU":
                target["recommendations"] = [
                    "Reduce embedding_dim from 512 to 256",
                    "Reduce num_layers from 3 to 2",
                    "Reduce hidden_dim from 1024 to 512",
                    "Reduce num_capsules from 8 to 4"
                ]
            elif component_name == "BCM":
                target["recommendations"] = [
                    "Reduce memory_size from 256 to 128",
                    "Reduce embedding_dim from 512 to 256",
                    "Reduce attention_heads from 4 to 2"
                ]
            elif component_name == "IGPM":
                target["recommendations"] = [
                    "Reduce plastic_slots from 32 to 16",
                    "Reduce max_episodic_memories from 500 to 250",
                    "Reduce instruction_embedding_dim from 128 to 64"
                ]
            elif component_name == "Embeddings":
                target["recommendations"] = [
                    "Reduce vocab_size from 50000 to 25000",
                    "Reduce embedding_dim from 512 to 256"
                ]
            elif component_name == "Position Embeddings":
                target["recommendations"] = [
                    "Use sinusoidal encoding instead of learned embeddings",
                    "Reduce max_sequence_length from 4096 to 2048"
                ]
            elif component_name == "MLCS":
                target["recommendations"] = [
                    "Increase compression_ratio from 0.05 to 0.1",
                    "Reduce codebook_size from 4096 to 2048",
                    "Reduce latent_dim from 128 to 64"
                ]
            
            targets.append(target)
    
    return targets


def generate_optimization_plan(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a detailed optimization plan to reach the 53M target.
    
    Args:
        analysis: Parameter analysis results
        
    Returns:
        Optimization plan with specific actions
    """
    targets = identify_optimization_targets(analysis)
    
    plan = {
        "current_state": {
            "total_parameters": analysis["total_parameters"],
            "target_parameters": 53_000_000,
            "reduction_needed": analysis["parameter_efficiency"]["reduction_needed"]
        },
        "optimization_targets": targets,
        "recommended_actions": [],
        "estimated_savings": 0
    }
    
    # Prioritize optimizations by potential impact
    sorted_targets = sorted(targets, key=lambda x: x["potential_reduction"], reverse=True)
    
    remaining_reduction = analysis["parameter_efficiency"]["reduction_needed"]
    total_savings = 0
    
    for target in sorted_targets:
        if remaining_reduction > 0 and target["potential_reduction"] > 0:
            # Calculate achievable reduction (don't exceed remaining need)
            achievable_reduction = min(target["potential_reduction"], remaining_reduction)
            
            action = {
                "component": target["component"],
                "current_parameters": target["current_parameters"],
                "target_parameters": target["current_parameters"] - achievable_reduction,
                "reduction": achievable_reduction,
                "recommendations": target["recommendations"]
            }
            
            plan["recommended_actions"].append(action)
            total_savings += achievable_reduction
            remaining_reduction -= achievable_reduction
    
    plan["estimated_savings"] = total_savings
    plan["remaining_after_optimization"] = analysis["total_parameters"] - total_savings
    
    return plan


def print_analysis_report(analysis: Dict[str, Any], detailed: bool = False):
    """
    Print a formatted analysis report.
    
    Args:
        analysis: Parameter analysis results
        detailed: Whether to print detailed module breakdown
    """
    print(f"\n🔍 PARAMETER ANALYSIS REPORT")
    print(f"=" * 60)
    
    # Summary
    total_params = analysis["total_parameters"]
    target_params = 53_000_000
    efficiency = analysis["parameter_efficiency"]
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Target Parameters: {target_params:,} ({target_params/1e6:.1f}M)")
    print(f"   Reduction Needed: {efficiency['reduction_needed']:,} ({efficiency['reduction_percentage']:.1f}%)")
    print(f"   Efficiency Ratio: {efficiency['efficiency_ratio']:.3f}")
    
    # Component breakdown
    print(f"\n🏗️ COMPONENT BREAKDOWN:")
    sorted_components = sorted(
        analysis["component_breakdown"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for component, params in sorted_components:
        percentage = (params / total_params) * 100
        print(f"   {component:20} {params:10,} ({percentage:5.1f}%)")
    
    # Largest modules
    if detailed:
        print(f"\n🔧 LARGEST MODULES:")
        for module_name, params in analysis["largest_modules"][:10]:
            percentage = (params / total_params) * 100
            print(f"   {module_name:40} {params:10,} ({percentage:5.1f}%)")
    
    # Optimization targets
    targets = identify_optimization_targets(analysis)
    print(f"\n🎯 OPTIMIZATION TARGETS:")
    for target in targets[:5]:  # Top 5 targets
        print(f"   {target['component']:20} {target['current_parameters']:10,} → {target['potential_reduction']:10,} reduction")
        for rec in target['recommendations'][:2]:  # Top 2 recommendations
            print(f"     • {rec}")
    
    # Optimization plan
    plan = generate_optimization_plan(analysis)
    print(f"\n📋 OPTIMIZATION PLAN:")
    print(f"   Estimated Savings: {plan['estimated_savings']:,} parameters")
    print(f"   Remaining After Optimization: {plan['remaining_after_optimization']:,} parameters")
    print(f"   Target Achievement: {(plan['remaining_after_optimization'] / target_params) * 100:.1f}%")
    
    if plan['remaining_after_optimization'] <= target_params:
        print(f"   ✅ Target achievable with recommended optimizations")
    else:
        print(f"   ⚠️ Additional optimizations needed")


def save_analysis_results(analysis: Dict[str, Any], output_path: str):
    """
    Save analysis results to JSON file.
    
    Args:
        analysis: Parameter analysis results
        output_path: Path to save the results
    """
    # Add optimization plan to analysis
    analysis["optimization_plan"] = generate_optimization_plan(analysis)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n💾 Analysis results saved to: {output_path}")


def main():
    """Main function for parameter analysis."""
    parser = argparse.ArgumentParser(description="Analyze CORE-NN Parameter Counts")
    parser.add_argument("--config", type=str, default="configs/laptop_optimized.yaml",
                       help="Configuration file path")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed module breakdown")
    parser.add_argument("--output", type=str, default="analysis/parameter_analysis_results.json",
                       help="Output file for analysis results")
    parser.add_argument("--target", type=int, default=53_000_000,
                       help="Target parameter count (default: 53M)")
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing parameter counts for CORE-NN")
    print(f"Config: {args.config}")
    print(f"Target: {args.target:,} parameters")
    
    try:
        # Load configuration and model
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Try to use extended model if available
        try:
            from optimization.position_embedding_fix import ExtendedFlexibleCoreNNModel
            model = ExtendedFlexibleCoreNNModel(config, max_sequence_length=4096)
            print("Using extended flexible sequence length model")
        except ImportError:
            try:
                from optimization.flexible_sequence_handling import FlexibleCoreNNModel
                model = FlexibleCoreNNModel(config, max_sequence_length=200)
                print("Using flexible sequence length model")
            except ImportError:
                model = CoreNNModel(config)
                print("Using standard model")
        
        model.eval()
        
        # Perform analysis
        print(f"\n📊 Performing parameter analysis...")
        analysis = analyze_component_parameters(model)
        
        # Print report
        print_analysis_report(analysis, detailed=args.detailed)
        
        # Save results
        save_analysis_results(analysis, args.output)
        
        # Summary
        total_params = analysis["total_parameters"]
        target_params = args.target
        efficiency = analysis["parameter_efficiency"]
        
        print(f"\n✅ Parameter analysis completed!")
        print(f"   Current: {total_params:,} parameters ({total_params/1e6:.1f}M)")
        print(f"   Target: {target_params:,} parameters ({target_params/1e6:.1f}M)")
        print(f"   Reduction needed: {efficiency['reduction_needed']:,} parameters")
        
        if efficiency['reduction_needed'] > 0:
            print(f"   ⚠️ Parameter reduction required to meet target")
        else:
            print(f"   ✅ Target already achieved")
        
    except Exception as e:
        print(f"❌ Error during parameter analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 