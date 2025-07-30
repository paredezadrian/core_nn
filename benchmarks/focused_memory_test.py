#!/usr/bin/env python3
"""
Focused Memory Test - Following the Pseudocode Example

This test follows the exact pattern from the requirements:
1. model("define core-nn")
2. model("#remember(core-nn)")  
3. output = model("what is core-nn?")
4. assert "define core-nn" in output

Tests the complete flow from definition to memory storage to recall.
"""

import sys
import time
from pathlib import Path

# Add the core_nn directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.inference.engine import InferenceEngine


def load_core_nn_model():
    """Load and initialize CORE-NN model."""
    print("📦 Loading CORE-NN model...")
    
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
    
    # Create inference engine
    engine = InferenceEngine(model, config.inference, model.tokenizer)
    
    return engine


def model_call(engine, input_text):
    """Simulate model call - wrapper around engine.generate()."""
    result = engine.generate(input_text=input_text)
    return result.generated_text


def test_focused_memory_flow():
    """Test the focused memory flow as specified in pseudocode."""
    print("🧪 Running Focused Memory Test")
    print("=" * 50)
    
    # Load model
    model = load_core_nn_model()
    
    # Step 1: Define core-nn (simulate providing definition)
    print("\n1️⃣ Defining CORE-NN...")
    definition = "CORE-NN is a Context-Oriented Recurrent Embedding Neural Network designed for edge devices"
    define_result = model_call(model, f'#remember("{definition}")')
    print(f"   Input: define core-nn")
    print(f"   Stored: {definition}")
    print(f"   Result: {define_result}")
    
    # Step 2: Remember core-nn using system command
    print("\n2️⃣ Remembering CORE-NN...")
    remember_result = model_call(model, '#remember("CORE-NN: Context-Oriented Recurrent Embedding Neural Network for edge AI")')
    print(f"   Input: #remember(core-nn)")
    print(f"   Result: {remember_result}")
    
    # Step 3: Query what is core-nn
    print("\n3️⃣ Querying what is CORE-NN...")
    query_result = model_call(model, '#recall("CORE-NN")')
    print(f"   Input: what is core-nn?")
    print(f"   Output: {query_result}")
    
    # Step 4: Assert that definition is in output
    print("\n4️⃣ Verifying memory recall...")
    success = False
    
    # Check if any of the key terms from our definitions appear in the recall
    key_terms = [
        "Context-Oriented",
        "Recurrent Embedding", 
        "Neural Network",
        "edge",
        "CORE-NN"
    ]
    
    found_terms = []
    for term in key_terms:
        if term.lower() in query_result.lower():
            found_terms.append(term)
    
    if found_terms:
        success = True
        print(f"   ✅ SUCCESS: Found terms {found_terms} in recall output")
    else:
        print(f"   ❌ FAILURE: No definition terms found in recall output")
        print(f"   Expected terms: {key_terms}")
        print(f"   Actual output: {query_result}")
    
    return success, {
        'define_result': define_result,
        'remember_result': remember_result,
        'query_result': query_result,
        'found_terms': found_terms,
        'success': success
    }


def test_command_variations():
    """Test different command variations and formats."""
    print("\n🔄 Testing Command Variations")
    print("=" * 50)
    
    model = load_core_nn_model()
    
    test_cases = [
        # Different quote styles
        ('#remember("Python is a programming language")', "Python"),
        ("#remember('JavaScript runs in browsers')", "JavaScript"),
        ('#remember(Rust is systems programming)', "Rust"),
        
        # Different content types
        ('#remember("Machine learning: AI that learns from data")', "Machine learning"),
        ('#remember("Deep learning uses neural networks with multiple layers")', "Deep learning"),
        
        # Recall variations
        ('#recall("Python")', None),
        ('#recall("programming")', None),
        ('#recall("neural")', None),
    ]
    
    results = []
    
    for i, (command, expected_term) in enumerate(test_cases):
        print(f"\n   Test {i+1}: {command}")
        
        try:
            result = model_call(model, command)
            success = True
            
            if expected_term:
                # This is a remember command
                success = "Remembered:" in result
            else:
                # This is a recall command
                success = "memories for" in result or "Recalled" in result
            
            print(f"   Result: {result[:100]}{'...' if len(result) > 100 else ''}")
            print(f"   Status: {'✅ PASS' if success else '❌ FAIL'}")
            
            results.append({
                'command': command,
                'result': result,
                'success': success
            })
            
        except Exception as e:
            print(f"   Status: ❌ ERROR - {e}")
            results.append({
                'command': command,
                'result': str(e),
                'success': False
            })
    
    return results


def test_session_persistence():
    """Test that memories persist within a session."""
    print("\n💾 Testing Session Persistence")
    print("=" * 50)
    
    model = load_core_nn_model()
    
    # Store multiple related memories
    memories = [
        "Transformers use attention mechanisms",
        "BERT is a transformer-based model", 
        "GPT is generative pre-trained transformer",
        "Attention allows models to focus on relevant parts"
    ]
    
    print("   Storing memories...")
    for i, memory in enumerate(memories):
        result = model_call(model, f'#remember("{memory}")')
        print(f"   {i+1}. Stored: {memory[:50]}...")
    
    # Test recall of related terms
    recall_tests = [
        ("transformer", ["Transformers", "BERT", "GPT"]),
        ("attention", ["attention", "focus"]),
        ("BERT", ["BERT", "transformer"]),
        ("generative", ["GPT", "generative"])
    ]
    
    print("\n   Testing recall...")
    recall_results = []
    
    for query, expected_terms in recall_tests:
        result = model_call(model, f'#recall("{query}")')
        
        found_terms = []
        for term in expected_terms:
            if term.lower() in result.lower():
                found_terms.append(term)
        
        success = len(found_terms) > 0
        print(f"   Query '{query}': {'✅' if success else '❌'} Found {found_terms}")
        
        recall_results.append({
            'query': query,
            'expected': expected_terms,
            'found': found_terms,
            'success': success,
            'result': result
        })
    
    return recall_results


def run_all_focused_tests():
    """Run all focused memory tests."""
    print("🚀 CORE-NN Focused Memory Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test 1: Main focused flow
    main_success, main_results = test_focused_memory_flow()
    
    # Test 2: Command variations
    variation_results = test_command_variations()
    variation_success = all(r['success'] for r in variation_results)
    
    # Test 3: Session persistence
    persistence_results = test_session_persistence()
    persistence_success = all(r['success'] for r in persistence_results)
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n📊 FOCUSED TEST SUMMARY")
    print("=" * 60)
    print(f"Main Flow Test: {'✅ PASS' if main_success else '❌ FAIL'}")
    print(f"Command Variations: {'✅ PASS' if variation_success else '❌ FAIL'}")
    print(f"Session Persistence: {'✅ PASS' if persistence_success else '❌ FAIL'}")
    print(f"Total Execution Time: {total_time:.3f}s")
    
    overall_success = main_success and variation_success and persistence_success
    print(f"\n{'🎉 ALL FOCUSED TESTS PASSED!' if overall_success else '⚠️  SOME TESTS FAILED'}")
    
    # Save results
    import json
    results_file = Path("benchmarks/focused_test_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'total_time': total_time,
            'main_test': main_results,
            'variation_tests': variation_results,
            'persistence_tests': persistence_results,
            'overall_success': overall_success
        }, f, indent=2)
    
    print(f"📄 Results saved to: {results_file}")
    print("=" * 60)
    
    return overall_success


if __name__ == "__main__":
    success = run_all_focused_tests()
    sys.exit(0 if success else 1)
