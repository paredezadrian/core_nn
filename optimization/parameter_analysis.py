"""
Parameter Analysis Tool for CORE-NN Optimization.

This module provides comprehensive analysis of parameter distribution
across CORE-NN components to identify optimization opportunities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys
import json
from dataclasses import dataclass, asdict

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.config.schema import CoreNNConfig


@dataclass
class ComponentAnalysis:
    """Analysis results for a single component."""
    name: str
    total_parameters: int
    trainable_parameters: int
    parameter_percentage: float
    memory_usage_mb: float
    subcomponents: Dict[str, int]


@dataclass
class ParameterAnalysisResult:
    """Complete parameter analysis results."""
    total_parameters: int
    trainable_parameters: int
    component_breakdown: List[ComponentAnalysis]
    optimization_opportunities: List[str]
    efficiency_score: float


class ParameterAnalyzer:
    """Analyzes parameter distribution in CORE-NN model."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = ConfigManager().load_config(config_path)
        self.model = CoreNNModel(self.config, vocab_size=50000)
        self.model.eval()
        
        print(f"Loaded CORE-NN model for analysis")
        print(f"Total parameters: {self.get_total_parameters():,}")
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def analyze_component(self, component: nn.Module, name: str, total_params: int) -> ComponentAnalysis:
        """Analyze a single component."""
        component_params = sum(p.numel() for p in component.parameters())
        trainable_params = sum(p.numel() for p in component.parameters() if p.requires_grad)
        percentage = (component_params / total_params) * 100
        
        # Estimate memory usage (rough approximation)
        memory_mb = (component_params * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Analyze subcomponents
        subcomponents = {}
        for sub_name, sub_module in component.named_children():
            sub_params = sum(p.numel() for p in sub_module.parameters())
            if sub_params > 0:
                subcomponents[sub_name] = sub_params
        
        return ComponentAnalysis(
            name=name,
            total_parameters=component_params,
            trainable_parameters=trainable_params,
            parameter_percentage=percentage,
            memory_usage_mb=memory_mb,
            subcomponents=subcomponents
        )
    
    def run_analysis(self) -> ParameterAnalysisResult:
        """Run comprehensive parameter analysis."""
        print("Running comprehensive parameter analysis...")
        
        total_params = self.get_total_parameters()
        trainable_params = self.get_trainable_parameters()
        
        # Analyze major components
        components = []
        
        # 1. Token Embedding
        if hasattr(self.model, 'token_embedding'):
            components.append(self.analyze_component(
                self.model.token_embedding, "Token Embedding", total_params
            ))
        
        # 2. Position Embedding
        if hasattr(self.model, 'position_embedding'):
            components.append(self.analyze_component(
                self.model.position_embedding, "Position Embedding", total_params
            ))
        
        # 3. BCM (Biological Core Memory)
        if hasattr(self.model, 'bcm'):
            components.append(self.analyze_component(
                self.model.bcm, "BCM (Biological Core Memory)", total_params
            ))
        
        # 4. RTEU (Recursive Temporal Embedding Unit)
        if hasattr(self.model, 'rteu'):
            components.append(self.analyze_component(
                self.model.rteu, "RTEU (Recursive Temporal Embedding)", total_params
            ))
        
        # 5. IGPM (Instruction-Guided Plasticity Module)
        if hasattr(self.model, 'igpm'):
            components.append(self.analyze_component(
                self.model.igpm, "IGPM (Instruction-Guided Plasticity)", total_params
            ))
        
        # 6. MLCS (Multi-Level Compression System)
        if hasattr(self.model, 'mlcs'):
            components.append(self.analyze_component(
                self.model.mlcs, "MLCS (Multi-Level Compression)", total_params
            ))
        
        # 7. Output layers
        output_params = 0
        output_components = {}
        for name, module in self.model.named_children():
            if 'output' in name.lower() or 'head' in name.lower() or 'classifier' in name.lower():
                params = sum(p.numel() for p in module.parameters())
                output_params += params
                output_components[name] = params
        
        if output_params > 0:
            components.append(ComponentAnalysis(
                name="Output Layers",
                total_parameters=output_params,
                trainable_parameters=output_params,
                parameter_percentage=(output_params / total_params) * 100,
                memory_usage_mb=(output_params * 4) / (1024 * 1024),
                subcomponents=output_components
            ))
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(components, total_params)
        
        # Calculate efficiency score (lower is better)
        efficiency_score = self._calculate_efficiency_score(components)
        
        return ParameterAnalysisResult(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            component_breakdown=components,
            optimization_opportunities=optimization_opportunities,
            efficiency_score=efficiency_score
        )
    
    def _identify_optimization_opportunities(self, components: List[ComponentAnalysis], total_params: int) -> List[str]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # Sort components by parameter count
        sorted_components = sorted(components, key=lambda x: x.total_parameters, reverse=True)
        
        for component in sorted_components:
            if component.parameter_percentage > 20:
                opportunities.append(
                    f"HIGH PRIORITY: {component.name} uses {component.parameter_percentage:.1f}% "
                    f"({component.total_parameters:,} params) - major optimization target"
                )
            elif component.parameter_percentage > 10:
                opportunities.append(
                    f"MEDIUM PRIORITY: {component.name} uses {component.parameter_percentage:.1f}% "
                    f"({component.total_parameters:,} params) - optimization opportunity"
                )
        
        # Check for potential parameter sharing opportunities
        igpm_component = next((c for c in components if "IGPM" in c.name), None)
        if igpm_component and igpm_component.parameter_percentage > 15:
            opportunities.append(
                "PARAMETER SHARING: IGPM likely has redundant parameters across slots - "
                "implement parameter sharing"
            )
        
        # Check for embedding efficiency
        embedding_total = sum(c.total_parameters for c in components 
                            if "Embedding" in c.name)
        if embedding_total / total_params > 0.1:
            opportunities.append(
                f"EMBEDDING OPTIMIZATION: Embeddings use {(embedding_total/total_params)*100:.1f}% "
                f"of parameters - consider compression or sharing"
            )
        
        return opportunities
    
    def _calculate_efficiency_score(self, components: List[ComponentAnalysis]) -> float:
        """Calculate efficiency score (0-100, lower is better)."""
        # Penalize components with high parameter usage
        score = 0.0
        
        for component in components:
            # Penalize high parameter usage
            if component.parameter_percentage > 30:
                score += 30
            elif component.parameter_percentage > 20:
                score += 20
            elif component.parameter_percentage > 10:
                score += 10
        
        # Bonus for balanced distribution
        percentages = [c.parameter_percentage for c in components]
        if len(percentages) > 1:
            variance = sum((p - sum(percentages)/len(percentages))**2 for p in percentages) / len(percentages)
            score += variance / 10  # Penalize high variance
        
        return min(score, 100.0)
    
    def print_analysis(self, result: ParameterAnalysisResult):
        """Print detailed analysis results."""
        print("\n" + "="*60)
        print("CORE-NN PARAMETER ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Parameters: {result.total_parameters:,}")
        print(f"  Trainable Parameters: {result.trainable_parameters:,}")
        print(f"  Efficiency Score: {result.efficiency_score:.1f}/100 (lower is better)")
        
        print(f"\nCOMPONENT BREAKDOWN:")
        sorted_components = sorted(result.component_breakdown, 
                                 key=lambda x: x.total_parameters, reverse=True)
        
        for component in sorted_components:
            print(f"\n  {component.name}:")
            print(f"    Parameters: {component.total_parameters:,} ({component.parameter_percentage:.1f}%)")
            print(f"    Memory: {component.memory_usage_mb:.1f} MB")
            
            if component.subcomponents:
                print(f"    Subcomponents:")
                for sub_name, sub_params in sorted(component.subcomponents.items(), 
                                                 key=lambda x: x[1], reverse=True):
                    sub_percentage = (sub_params / result.total_parameters) * 100
                    print(f"      {sub_name}: {sub_params:,} ({sub_percentage:.1f}%)")
        
        print(f"\nOPTIMIZATION OPPORTUNITIES:")
        for i, opportunity in enumerate(result.optimization_opportunities, 1):
            print(f"  {i}. {opportunity}")
        
        print("\n" + "="*60)
    
    def save_analysis(self, result: ParameterAnalysisResult, output_path: str):
        """Save analysis results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"\nAnalysis saved to: {output_file}")


def main():
    """Run parameter analysis."""
    analyzer = ParameterAnalyzer()
    result = analyzer.run_analysis()
    
    # Print results
    analyzer.print_analysis(result)
    
    # Save results
    analyzer.save_analysis(result, "optimization/results/parameter_analysis.json")
    
    return result


if __name__ == "__main__":
    main()
