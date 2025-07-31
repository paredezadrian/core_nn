"""
Configuration utilities for CORE-NN.

Provides utility functions for configuration loading, merging, and processing.
"""

import os
import yaml
import json
import re
from pathlib import Path
from typing import Dict, Any, Union, Optional
from copy import deepcopy


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict containing configuration data
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            return yaml.safe_load(f)


def save_config(config_dict: Dict[str, Any], output_path: Union[str, Path]):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config_dict: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = deepcopy(value)
    
    return merged


def resolve_env_vars(config_dict: Dict[str, Any], 
                    env_prefix: str = "CORE_NN_") -> Dict[str, Any]:
    """
    Resolve environment variables in configuration.
    
    Supports syntax like ${ENV_VAR} or ${ENV_VAR:default_value}
    
    Args:
        config_dict: Configuration dictionary
        env_prefix: Environment variable prefix
        
    Returns:
        Configuration with resolved environment variables
    """
    def resolve_value(value):
        if isinstance(value, str):
            # Pattern to match ${VAR} or ${VAR:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_env_var(match):
                env_var = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                
                # Get environment variable value
                env_value = os.getenv(env_var, default_value)
                
                # Try to convert to appropriate type
                return convert_string_value(env_value)
            
            return re.sub(pattern, replace_env_var, value)
        
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        
        else:
            return value
    
    return resolve_value(config_dict)


def convert_string_value(value: str) -> Union[str, int, float, bool, None]:
    """
    Convert string value to appropriate Python type.
    
    Args:
        value: String value to convert
        
    Returns:
        Converted value
    """
    if not isinstance(value, str):
        return value
    
    # Handle None/null
    if value.lower() in ('none', 'null', ''):
        return None
    
    # Handle boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Handle integer
    try:
        if '.' not in value and 'e' not in value.lower():
            return int(value)
    except ValueError:
        pass
    
    # Handle float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def validate_config_structure(config_dict: Dict[str, Any], 
                            required_sections: Optional[list] = None) -> list:
    """
    Validate basic configuration structure.
    
    Args:
        config_dict: Configuration dictionary to validate
        required_sections: List of required top-level sections
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if required_sections is None:
        required_sections = ['model', 'bcm', 'rteu', 'igpm', 'mlcs', 'execution_engine']
    
    # Check required sections
    for section in required_sections:
        if section not in config_dict:
            errors.append(f"Missing required section: {section}")
        elif not isinstance(config_dict[section], dict):
            errors.append(f"Section '{section}' must be a dictionary")
    
    return errors


def get_config_diff(config1: Dict[str, Any], 
                   config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get differences between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary showing differences
    """
    def compare_values(val1, val2, path=""):
        if isinstance(val1, dict) and isinstance(val2, dict):
            diff = {}
            all_keys = set(val1.keys()) | set(val2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in val1:
                    diff[key] = {"status": "added", "value": val2[key]}
                elif key not in val2:
                    diff[key] = {"status": "removed", "value": val1[key]}
                else:
                    subdiff = compare_values(val1[key], val2[key], current_path)
                    if subdiff:
                        diff[key] = subdiff
            
            return diff if diff else None
        
        elif val1 != val2:
            return {
                "status": "changed",
                "old_value": val1,
                "new_value": val2
            }
        
        return None
    
    return compare_values(config1, config2) or {}


def flatten_config(config_dict: Dict[str, Any], 
                  separator: str = ".") -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.
    
    Args:
        config_dict: Nested configuration dictionary
        separator: Separator for flattened keys
        
    Returns:
        Flattened configuration dictionary
    """
    def flatten_recursive(obj, parent_key=""):
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(flatten_recursive(value, new_key).items())
        else:
            return {parent_key: obj}
        
        return dict(items)
    
    return flatten_recursive(config_dict)


def unflatten_config(flat_config: Dict[str, Any], 
                    separator: str = ".") -> Dict[str, Any]:
    """
    Unflatten configuration dictionary.
    
    Args:
        flat_config: Flattened configuration dictionary
        separator: Separator used in flattened keys
        
    Returns:
        Nested configuration dictionary
    """
    result = {}
    
    for key, value in flat_config.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def create_config_template(template_name: str = "basic") -> Dict[str, Any]:
    """
    Create a configuration template.
    
    Args:
        template_name: Name of template to create
        
    Returns:
        Configuration template dictionary
    """
    if template_name == "basic":
        return {
            "model": {
                "name": "core-nn-basic",
                "version": "0.2.2"
            },
            "bcm": {
                "memory_size": 256,
                "embedding_dim": 512,
                "salience_threshold": 0.7
            },
            "rteu": {
                "num_layers": 2,
                "embedding_dim": 512,
                "hidden_dim": 1024
            },
            "igpm": {
                "plastic_slots": 32,
                "meta_learning_rate": 0.001
            },
            "mlcs": {
                "compression_ratio": 0.1,
                "latent_dim": 128
            },
            "execution_engine": {
                "memory_budget_gb": 8,
                "max_concurrent_modules": 2
            },
            "device": {
                "preferred": "auto",
                "mixed_precision": True
            }
        }
    
    elif template_name == "minimal":
        return {
            "model": {"name": "core-nn-minimal", "version": "0.2.2"},
            "bcm": {"memory_size": 128, "embedding_dim": 256},
            "rteu": {"num_layers": 1, "embedding_dim": 256},
            "igpm": {"plastic_slots": 16},
            "mlcs": {"compression_ratio": 0.05},
            "execution_engine": {"memory_budget_gb": 4},
            "device": {"preferred": "cpu"}
        }
    
    else:
        raise ValueError(f"Unknown template: {template_name}")


def export_config_schema() -> Dict[str, Any]:
    """
    Export configuration schema for documentation or validation.
    
    Returns:
        Configuration schema dictionary
    """
    return {
        "type": "object",
        "properties": {
            "model": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"}
                },
                "required": ["name", "version"]
            },
            "bcm": {
                "type": "object",
                "properties": {
                    "memory_size": {"type": "integer", "minimum": 1},
                    "embedding_dim": {"type": "integer", "minimum": 1},
                    "salience_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "rteu": {
                "type": "object",
                "properties": {
                    "num_layers": {"type": "integer", "minimum": 1},
                    "embedding_dim": {"type": "integer", "minimum": 1},
                    "hidden_dim": {"type": "integer", "minimum": 1}
                }
            },
            "device": {
                "type": "object",
                "properties": {
                    "preferred": {"type": "string", "enum": ["auto", "cpu", "cuda", "mps"]},
                    "mixed_precision": {"type": "boolean"}
                }
            }
        },
        "required": ["model", "bcm", "rteu", "igpm", "mlcs", "execution_engine"]
    }
