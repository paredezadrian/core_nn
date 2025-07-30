"""
Configuration Management for CORE-NN

This module provides configuration management including:
- YAML/JSON configuration loading and validation
- Schema definitions and validation
- Environment variable overrides
- Configuration merging and inheritance
"""

from .manager import ConfigManager
from .schema import CoreNNConfig, validate_config
from .utils import load_config, save_config, merge_configs

__all__ = [
    "ConfigManager",
    "CoreNNConfig", 
    "validate_config",
    "load_config",
    "save_config", 
    "merge_configs",
]
