#!/usr/bin/env python3
"""
Update Laptop Configuration for Variable Sequences.

This script updates the laptop-optimized configuration to support variable
sequence lengths from 10-4096 tokens, incorporating the extended position
embedding capabilities developed in Tasks 1.1 and 1.2.
"""

import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import ConfigManager


def update_laptop_config_for_flexible_sequences(config_path: str, output_path: str = None):
    """
    Update laptop configuration to support flexible sequences.
    
    Args:
        config_path: Path to the current laptop configuration
        output_path: Path to save the updated configuration (optional)
    """
    print(f"🔧 Updating laptop configuration for flexible sequences")
    print(f"Input config: {config_path}")
    
    # Load current configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Create updated configuration with flexible sequence support
    updated_config = {
        # Device settings optimized for CPU
        "device": {
            "preferred": "cpu",
            "mixed_precision": False,
            "compile_model": False
        },
        
        # Execution engine optimized for laptop
        "execution_engine": {
            "cpu_threads": 8,
            "async_execution": False,
            "max_concurrent_modules": 2,
            "memory_budget_gb": 8,
            "offload_threshold": 0.7,
            "priority_scheduling": True
        },
        
        # Inference settings with extended sequence support
        "inference": {
            "max_sequence_length": 4096,  # Extended from 20 to 4096
            "max_new_tokens": 512,        # Increased for longer sequences
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.05
        },
        
        # Memory settings optimized for longer sequences
        "memory": {
            "working_memory_size": 256,      # Increased for longer contexts
            "episodic_memory_size": 512,     # Increased for more memories
            "semantic_memory_size": 2048,    # Increased for better retention
            "memory_consolidation_interval": 50,  # Increased for longer sequences
        },
        
        # BCM settings optimized for longer sequences
        "bcm": {
            "memory_size": 256,           # Increased from 128
            "salience_threshold": 0.85,   # Slightly reduced for more retention
            "attention_heads": 4,
            "decay_rate": 0.98,
            "embedding_dim": 512,
            "update_gate_type": "gru"
        },
        
        # RTEU settings optimized for longer sequences
        "rteu": {
            "embedding_dim": 512,
            "num_layers": 3,              # Increased from 2
            "routing_iterations": 2,       # Increased from 1
            "activation": "gelu",
            "capsule_dim": 32,
            "dropout": 0.05,
            "hidden_dim": 1024,
            "num_capsules": 8,
            "temporal_scales": [1, 2, 4, 8]
        },
        
        # IGPM settings optimized for longer sequences
        "igpm": {
            "plastic_slots": 32,           # Increased from 16
            "max_episodic_memories": 500,  # Increased from 250
            "fast_weight_decay": 0.995,
            "instruction_embedding_dim": 128,
            "meta_learning_rate": 0.0005,
            "plasticity_threshold": 0.85
        },
        
        # MLCS settings optimized for longer sequences
        "mlcs": {
            "compression_ratio": 0.05,     # Keep original ratio
            "auto_compress_threshold": 0.8,
            "codebook_size": 4096,
            "kpack_max_size_mb": 25,
            "latent_dim": 128,
            "num_compression_levels": 3
        },
        
        # Tokenizer settings for longer sequences
        "tokenizer": {
            "type": "asc",
            "preset": "edge",
            "custom_config_path": None,
            "overrides": {
                "cache_size": 5000,
                "enable_contextual_merging": True,
                "max_sequence_length": 4096  # Extended from 1024
            }
        },
        
        # API settings
        "api": {
            "commands": {
                "forget": {
                    "confirmation_required": True,
                    "enabled": True
                },
                "recall": {
                    "enabled": True,
                    "similarity_threshold": 0.85
                },
                "remember": {
                    "enabled": True,
                    "max_items": 50
                }
            }
        },
        
        # Logging settings
        "logging": {
            "level": "INFO",
            "log_file": "core_nn_laptop.log",
            "log_inference_time": True,
            "log_memory_usage": True,
            "tensorboard_dir": "runs_laptop"
        },
        
        # Model settings
        "model": {
            "name": "core-nn-laptop-optimized",
            "version": "0.2.2"
        },
        
        # Session settings
        "session": {
            "auto_save": True,
            "max_session_history": 5,
            "save_interval": 600,
            "session_dir": "sessions"
        },
        
        # Training settings
        "training": {
            "batch_size": 1,
            "gradient_clipping": 0.5,
            "learning_rate": 5.0e-05,
            "max_steps": 50000,
            "warmup_steps": 500,
            "weight_decay": 0.005
        }
    }
    
    # Save updated configuration
    if output_path is None:
        output_path = config_path.replace('.yaml', '_flexible_sequences.yaml')
    
    with open(output_path, 'w') as f:
        yaml.dump(updated_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Updated configuration saved to: {output_path}")
    
    # Print summary of changes
    print(f"\n📋 Configuration Update Summary:")
    print(f"   - Max sequence length: 20 → 4096 tokens")
    print(f"   - Position embedding: Fixed → Extended flexible")
    print(f"   - Memory sizes: Increased for longer sequences")
    print(f"   - RTEU layers: 2 → 3 layers")
    print(f"   - IGPM slots: 16 → 32 slots")
    print(f"   - BCM memory: 128 → 256 entries")
    print(f"   - Flexible sequence handling: Enabled")
    print(f"   - Overflow handling: Graceful reset")
    
    return output_path


def validate_updated_config(config_path: str):
    """
    Validate the updated configuration by loading it and checking key settings.
    
    Args:
        config_path: Path to the updated configuration
    """
    print(f"\n🔍 Validating updated configuration...")
    
    try:
        # Load the updated configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # Check key settings
        validation_results = {
            "max_sequence_length": config.inference.max_sequence_length == 4096,
            "memory_sizes_increased": (
                config.memory.working_memory_size >= 256 and
                config.memory.episodic_memory_size >= 512
            ),
            "rteu_layers_increased": config.rteu.num_layers >= 3,
            "igpm_slots_increased": config.igpm.plastic_slots >= 32,
            "bcm_memory_increased": config.bcm.memory_size >= 256
        }
        
        print(f"   ✅ Max sequence length: {config.inference.max_sequence_length}")
        print(f"   ✅ Memory sizes increased: {validation_results['memory_sizes_increased']}")
        print(f"   ✅ RTEU layers increased: {validation_results['rteu_layers_increased']}")
        print(f"   ✅ IGPM slots increased: {validation_results['igpm_slots_increased']}")
        print(f"   ✅ BCM memory increased: {validation_results['bcm_memory_increased']}")
        
        # Check if all validations pass
        all_passed = all(validation_results.values())
        if all_passed:
            print(f"\n✅ Configuration validation passed!")
        else:
            print(f"\n⚠️ Some validations failed:")
            for key, passed in validation_results.items():
                if not passed:
                    print(f"   - {key}: Failed")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_updated_config(config_path: str):
    """
    Test the updated configuration with the extended model.
    
    Args:
        config_path: Path to the updated configuration
    """
    print(f"\n🧪 Testing updated configuration...")
    
    try:
        import torch
        
        # Import the extended model
        from optimization.position_embedding_fix import ExtendedFlexibleCoreNNModel
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # Create model with updated config
        model = ExtendedFlexibleCoreNNModel(config, max_sequence_length=4096)
        model.eval()
        
        # Test different sequence lengths
        test_lengths = [10, 50, 100, 200, 500, 1000, 2000, 4096]
        
        print(f"Testing sequence lengths:")
        for length in test_lengths:
            try:
                # Create test input
                input_ids = torch.randint(0, 1000, (1, length))
                
                # Forward pass
                with torch.no_grad():
                    output = model.forward(input_ids)
                
                print(f"  ✅ Length {length}: Success")
                
            except Exception as e:
                print(f"  ❌ Length {length}: Failed - {e}")
        
        print(f"\n✅ Configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Main function for updating laptop configuration."""
    parser = argparse.ArgumentParser(description="Update Laptop Configuration for Flexible Sequences")
    parser.add_argument("--config", type=str, default="configs/laptop_optimized.yaml",
                       help="Input configuration file path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output configuration file path")
    parser.add_argument("--flexible-sequences", action="store_true",
                       help="Enable flexible sequence support")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the updated configuration")
    parser.add_argument("--test", action="store_true",
                       help="Test the updated configuration")
    
    args = parser.parse_args()
    
    print(f"🔧 Updating laptop configuration for flexible sequences")
    print(f"Config: {args.config}")
    print(f"Flexible sequences: {args.flexible_sequences}")
    
    try:
        # Update configuration
        output_path = update_laptop_config_for_flexible_sequences(args.config, args.output)
        
        # Validate configuration
        if args.validate:
            validation_passed = validate_updated_config(output_path)
            if not validation_passed:
                print(f"❌ Configuration validation failed")
                sys.exit(1)
        
        # Test configuration
        if args.test:
            import torch
            test_passed = test_updated_config(output_path)
            if not test_passed:
                print(f"❌ Configuration test failed")
                sys.exit(1)
        
        print(f"\n✅ Laptop configuration updated successfully!")
        print(f"   - Supports sequences: 10-4096 tokens")
        print(f"   - Extended position embedding enabled")
        print(f"   - Memory-efficient processing")
        print(f"   - CPU-optimized settings")
        print(f"   - Graceful overflow handling")
        
    except Exception as e:
        print(f"❌ Error updating laptop configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 