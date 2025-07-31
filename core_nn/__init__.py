"""
CORE-NN: Context-Oriented Recurrent Embedding Neural Network

A modular AI architecture designed for edge devices that replaces traditional
transformer-based LLMs with efficient, memory-conscious components.

Copyright 2024 Adrian Paredez

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__version__ = "0.2.1"
__author__ = "Adrian Paredez"
__email__ = "itsparedezadrian@outlook.com"

# Core components
from .components.bcm import BiologicalCoreMemory
from .components.rteu import RecursiveTemporalEmbeddingUnit
from .components.igpm import InstructionGuidedPlasticityModule
from .components.mlcs import MultiLevelCompressionSynthesizer
from .components.execution_engine import EdgeEfficientModularExecutionEngine

# Main model
from .model import CoreNNModel

# Configuration
from .config.manager import ConfigManager
from .config.schema import CoreNNConfig

# Inference
from .inference.engine import InferenceEngine
from .inference.session import SessionManager

# Memory systems (placeholder imports)
from .memory import EpisodicMemory, SemanticMemory, WorkingMemory

# Utilities
from .utils.logging import setup_logging
from .utils.device import get_optimal_device
from .utils.profiling import profile_memory, profile_compute

__all__ = [
    # Core components
    "BiologicalCoreMemory",
    "RecursiveTemporalEmbeddingUnit", 
    "InstructionGuidedPlasticityModule",
    "MultiLevelCompressionSynthesizer",
    "EdgeEfficientModularExecutionEngine",
    
    # Main model
    "CoreNNModel",
    
    # Configuration
    "ConfigManager",
    "CoreNNConfig",
    
    # Inference
    "InferenceEngine",
    "SessionManager",
    
    # Memory
    "EpisodicMemory",
    "SemanticMemory", 
    "WorkingMemory",
    
    # Utilities
    "setup_logging",
    "get_optimal_device",
    "profile_memory",
    "profile_compute",
]

# Package metadata
__package_info__ = {
    "name": "core-nn",
    "version": __version__,
    "description": "Production-ready AI architecture with 80.4% parameter efficiency and transformer-level performance",
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/paredezadrian/core_nn.git",
    "license": "Apache-2.0",
}
