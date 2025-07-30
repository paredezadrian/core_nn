"""
Configuration Manager for CORE-NN.

Handles loading, validation, merging, and environment variable overrides
for CORE-NN configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import asdict

from .schema import CoreNNConfig, validate_config
from .utils import merge_configs, resolve_env_vars


class ConfigManager:
    """
    Manages CORE-NN configurations with support for:
    - YAML/JSON loading
    - Environment variable overrides
    - Configuration validation
    - Configuration merging and inheritance
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.logger = logging.getLogger(__name__)
        
        # Default configurations
        self.default_config = CoreNNConfig()
        
        # Environment variable prefix
        self.env_prefix = "CORE_NN_"
        
    def load_config(self, 
                   config_path: Union[str, Path],
                   base_config: Optional[str] = None,
                   validate: bool = True) -> CoreNNConfig:
        """
        Load configuration from file with optional base configuration.
        
        Args:
            config_path: Path to configuration file
            base_config: Optional base configuration to inherit from
            validate: Whether to validate the configuration
            
        Returns:
            CoreNNConfig: Loaded and validated configuration
        """
        config_path = Path(config_path)
        
        # Load base configuration if specified
        if base_config:
            base_config_path = self.config_dir / f"{base_config}.yaml"
            if base_config_path.exists():
                base_dict = self._load_config_file(base_config_path)
            else:
                self.logger.warning(f"Base config {base_config} not found, using default")
                base_dict = asdict(self.default_config)
        else:
            base_dict = asdict(self.default_config)
        
        # Load main configuration
        if config_path.exists():
            config_dict = self._load_config_file(config_path)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Merge configurations
        merged_dict = merge_configs(base_dict, config_dict)
        
        # Apply environment variable overrides
        merged_dict = self._apply_env_overrides(merged_dict)
        
        # Create configuration object
        config = CoreNNConfig.from_dict(merged_dict)
        
        # Validate if requested
        if validate:
            errors = validate_config(config)
            if errors:
                raise ValueError(f"Configuration validation failed: {errors}")
        
        return config
    
    def save_config(self, config: CoreNNConfig, output_path: Union[str, Path]):
        """Save configuration to file."""
        output_path = Path(output_path)
        
        # Convert to dictionary
        config_dict = config.to_dict()
        
        # Determine format from extension
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            # Default to YAML
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {output_path}")
    
    def create_config_from_template(self, 
                                   template_name: str,
                                   output_path: Union[str, Path],
                                   overrides: Optional[Dict[str, Any]] = None) -> CoreNNConfig:
        """
        Create configuration from template with optional overrides.
        
        Args:
            template_name: Name of template (e.g., 'edge_device', 'default')
            output_path: Where to save the new configuration
            overrides: Optional configuration overrides
            
        Returns:
            CoreNNConfig: Created configuration
        """
        # Load template
        template_path = self.config_dir / f"{template_name}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(f"Template {template_name} not found")
        
        config = self.load_config(template_path, validate=False)
        
        # Apply overrides
        if overrides:
            config_dict = config.to_dict()
            config_dict = merge_configs(config_dict, overrides)
            config = CoreNNConfig.from_dict(config_dict)
        
        # Validate
        errors = validate_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        # Save
        self.save_config(config, output_path)
        
        return config
    
    def get_available_configs(self) -> List[str]:
        """Get list of available configuration files."""
        configs = []
        
        for file_path in self.config_dir.glob("*.yaml"):
            configs.append(file_path.stem)
        
        for file_path in self.config_dir.glob("*.json"):
            configs.append(file_path.stem)
        
        return sorted(configs)
    
    def validate_config_file(self, config_path: Union[str, Path]) -> List[str]:
        """Validate a configuration file and return errors."""
        try:
            config = self.load_config(config_path, validate=False)
            return validate_config(config)
        except Exception as e:
            return [f"Failed to load configuration: {str(e)}"]
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config file {file_path}: {str(e)}")
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Get all environment variables with our prefix
        env_overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to config path
                config_key = key[len(self.env_prefix):].lower()
                
                # Convert CORE_NN_BCM_MEMORY_SIZE to bcm.memory_size
                key_parts = config_key.split('_')
                
                if len(key_parts) >= 2:
                    section = key_parts[0]
                    param = '_'.join(key_parts[1:])
                    
                    if section not in env_overrides:
                        env_overrides[section] = {}
                    
                    # Try to convert value to appropriate type
                    converted_value = self._convert_env_value(value)
                    env_overrides[section][param] = converted_value
        
        # Merge environment overrides
        if env_overrides:
            config_dict = merge_configs(config_dict, env_overrides)
            self.logger.info(f"Applied environment overrides: {list(env_overrides.keys())}")
        
        return config_dict
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get_config_summary(self, config: CoreNNConfig) -> Dict[str, Any]:
        """Get a summary of configuration settings."""
        return {
            "model": {
                "name": config.model.name,
                "version": config.model.version
            },
            "memory_settings": {
                "bcm_memory_size": config.bcm.memory_size,
                "plastic_slots": config.igpm.plastic_slots,
                "memory_budget_gb": config.execution_engine.memory_budget_gb
            },
            "performance_settings": {
                "rteu_layers": config.rteu.num_layers,
                "embedding_dim": config.rteu.embedding_dim,
                "max_concurrent_modules": config.execution_engine.max_concurrent_modules
            },
            "device_settings": {
                "preferred_device": config.device.preferred,
                "mixed_precision": config.device.mixed_precision,
                "memory_efficient": config.device.memory_efficient
            },
            "inference_settings": {
                "max_sequence_length": config.inference.max_sequence_length,
                "temperature": config.inference.temperature,
                "max_new_tokens": config.inference.max_new_tokens
            }
        }
    
    def create_deployment_config(self, 
                               base_config: CoreNNConfig,
                               deployment_type: str = "edge") -> CoreNNConfig:
        """
        Create optimized configuration for specific deployment.
        
        Args:
            base_config: Base configuration to optimize
            deployment_type: Type of deployment ('edge', 'server', 'mobile')
            
        Returns:
            CoreNNConfig: Optimized configuration
        """
        config_dict = base_config.to_dict()
        
        if deployment_type == "edge":
            # Optimize for edge devices
            optimizations = {
                "bcm": {
                    "memory_size": min(config_dict["bcm"]["memory_size"], 256),
                    "embedding_dim": min(config_dict["bcm"]["embedding_dim"], 512)
                },
                "rteu": {
                    "num_layers": min(config_dict["rteu"]["num_layers"], 2),
                    "hidden_dim": min(config_dict["rteu"]["hidden_dim"], 1024)
                },
                "execution_engine": {
                    "memory_budget_gb": min(config_dict["execution_engine"]["memory_budget_gb"], 8),
                    "max_concurrent_modules": min(config_dict["execution_engine"]["max_concurrent_modules"], 2)
                },
                "device": {
                    "preferred": "cpu",
                    "mixed_precision": False
                }
            }
        elif deployment_type == "server":
            # Optimize for server deployment
            optimizations = {
                "execution_engine": {
                    "memory_budget_gb": max(config_dict["execution_engine"]["memory_budget_gb"], 32),
                    "max_concurrent_modules": max(config_dict["execution_engine"]["max_concurrent_modules"], 8)
                },
                "device": {
                    "preferred": "cuda",
                    "mixed_precision": True,
                    "compile_model": True
                }
            }
        elif deployment_type == "mobile":
            # Optimize for mobile devices
            optimizations = {
                "bcm": {
                    "memory_size": min(config_dict["bcm"]["memory_size"], 128),
                    "embedding_dim": min(config_dict["bcm"]["embedding_dim"], 256)
                },
                "rteu": {
                    "num_layers": 1,
                    "hidden_dim": 512
                },
                "execution_engine": {
                    "memory_budget_gb": 4,
                    "max_concurrent_modules": 1,
                    "async_execution": False
                }
            }
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")
        
        # Merge optimizations
        optimized_dict = merge_configs(config_dict, optimizations)
        
        return CoreNNConfig.from_dict(optimized_dict)
