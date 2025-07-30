"""
CORE-NN Architecture Components

This module contains the five main architectural components of CORE-NN:

1. BiologicalCoreMemory (BCM) - Hippocampus-inspired temporal memory
2. RecursiveTemporalEmbeddingUnit (RTEU) - Attention replacement with temporal routing
3. InstructionGuidedPlasticityModule (IGPM) - Meta-learning without global updates
4. MultiLevelCompressionSynthesizer (MLCS) - Knowledge compression and .kpack management
5. EdgeEfficientModularExecutionEngine - Asynchronous modular execution
"""

from .bcm import BiologicalCoreMemory
from .rteu import RecursiveTemporalEmbeddingUnit
from .igpm import InstructionGuidedPlasticityModule
from .mlcs import MultiLevelCompressionSynthesizer
from .execution_engine import EdgeEfficientModularExecutionEngine

__all__ = [
    "BiologicalCoreMemory",
    "RecursiveTemporalEmbeddingUnit",
    "InstructionGuidedPlasticityModule", 
    "MultiLevelCompressionSynthesizer",
    "EdgeEfficientModularExecutionEngine",
]
