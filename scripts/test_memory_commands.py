#!/usr/bin/env python3
"""
Test script for memory commands (#remember, #recall, #forget) functionality.

This script tests the integration between the tokenizer, engine, and memory systems.
"""

import sys
import torch
from pathlib import Path

# Add the core_nn directory to the path
sys.path.insert(0, str(Path(__file__).parent / "core_nn"))

from core_nn import CoreNNModel, ConfigManager
from core_nn.inference.engine import InferenceEngine


def test_memory_commands():
    """Test the memory command functionality."""
    print("🧪 Testing CORE-NN Memory Commands")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        # Try to load default config file, fallback to default_config attribute
        try:
            default_config_path = Path("configs/default.yaml")
            if default_config_path.exists():
                config = config_manager.load_config(default_config_path)
            else:
                config = config_manager.default_config
        except Exception as e:
            print(f"Warning: Could not load config file, using default: {e}")
            config = config_manager.default_config
        
        # Create model
        print("📦 Creating CORE-NN model...")
        model = CoreNNModel(config)
        
        # Create inference engine
        print("🚀 Creating inference engine...")
        engine = InferenceEngine(model, config.inference, model.tokenizer)
        
        # Test #remember command
        print("\n🧠 Testing #remember command...")
        remember_text = '#remember("The capital of France is Paris")'
        result = engine.generate(input_text=remember_text)
        print(f"Input: {remember_text}")
        print(f"Output: {result.generated_text}")
        print(f"Component info: {result.component_info}")
        
        # Test another remember command
        remember_text2 = '#remember("Python is a programming language")'
        result2 = engine.generate(input_text=remember_text2)
        print(f"\nInput: {remember_text2}")
        print(f"Output: {result2.generated_text}")
        
        # Test #recall command
        print("\n🔍 Testing #recall command...")
        recall_text = '#recall("France")'
        result3 = engine.generate(input_text=recall_text)
        print(f"Input: {recall_text}")
        print(f"Output: {result3.generated_text}")
        print(f"Component info: {result3.component_info}")
        
        # Test #recall with different query
        recall_text2 = '#recall("Python")'
        result4 = engine.generate(input_text=recall_text2)
        print(f"\nInput: {recall_text2}")
        print(f"Output: {result4.generated_text}")
        
        # Test #forget command
        print("\n🗑️  Testing #forget command...")
        forget_text = '#forget("France")'
        result5 = engine.generate(input_text=forget_text)
        print(f"Input: {forget_text}")
        print(f"Output: {result5.generated_text}")
        print(f"Component info: {result5.component_info}")
        
        # Test recall after forget
        print("\n🔍 Testing recall after forget...")
        recall_after_forget = '#recall("France")'
        result6 = engine.generate(input_text=recall_after_forget)
        print(f"Input: {recall_after_forget}")
        print(f"Output: {result6.generated_text}")
        
        # Test episodic store stats
        print("\n📊 Episodic Store Statistics:")
        stats = engine.episodic_store.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ All memory command tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_tokenizer_integration():
    """Test that the tokenizer properly handles system commands."""
    print("\n🔧 Testing Tokenizer Integration")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        # Try to load default config file, fallback to default_config attribute
        try:
            default_config_path = Path("configs/default.yaml")
            if default_config_path.exists():
                config = config_manager.load_config(default_config_path)
            else:
                config = config_manager.default_config
        except Exception as e:
            print(f"Warning: Could not load config file, using default: {e}")
            config = config_manager.default_config
        
        # Create model
        model = CoreNNModel(config)
        tokenizer = model.tokenizer
        
        # Test system command tokenization
        test_commands = [
            '#remember("test content")',
            '#recall("test query")',
            '#forget("test term")',
            'Regular text without commands',
            '#remember(simple content without quotes)'
        ]
        
        for command in test_commands:
            print(f"\nTesting: {command}")
            tokens = tokenizer.tokenize(command, add_special_tokens=True)
            print(f"Tokens: {tokens}")
            
            # Check if system command tokens are present
            system_tokens = []
            for token_id in tokens:
                if hasattr(tokenizer, 'vocabulary'):
                    token = tokenizer.vocabulary.get_token(token_id)
                    if token and token.startswith('#'):
                        system_tokens.append(token)
            
            if system_tokens:
                print(f"System tokens found: {system_tokens}")
            else:
                print("No system tokens detected")
            
            # Test detokenization
            detokenized = tokenizer.detokenize(tokens, skip_special_tokens=False)
            print(f"Detokenized: {detokenized}")
        
        print("\n✅ Tokenizer integration tests completed!")
        
    except Exception as e:
        print(f"❌ Error during tokenizer testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_direct_model_methods():
    """Test the direct model memory methods for comparison."""
    print("\n🎯 Testing Direct Model Methods")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        # Try to load default config file, fallback to default_config attribute
        try:
            default_config_path = Path("configs/default.yaml")
            if default_config_path.exists():
                config = config_manager.load_config(default_config_path)
            else:
                config = config_manager.default_config
        except Exception as e:
            print(f"Warning: Could not load config file, using default: {e}")
            config = config_manager.default_config
        
        # Create model
        model = CoreNNModel(config)
        
        # Test direct remember
        print("Testing model.remember()...")
        result = model.remember("Direct memory: The sky is blue")
        print(f"Remember result: {result}")
        
        # Test direct recall
        print("\nTesting model.recall()...")
        memories = model.recall("sky")
        print(f"Recall result: {memories}")
        
        # Test direct forget
        print("\nTesting model.forget()...")
        forget_result = model.forget("sky")
        print(f"Forget result: {forget_result}")
        
        print("\n✅ Direct model method tests completed!")
        
    except Exception as e:
        print(f"❌ Error during direct model testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 Starting CORE-NN Memory Command Tests")
    print("=" * 60)
    
    success = True
    
    # Run all tests
    success &= test_memory_commands()
    success &= test_tokenizer_integration()
    success &= test_direct_model_methods()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed successfully!")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    print("=" * 60)
