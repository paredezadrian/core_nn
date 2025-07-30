#!/usr/bin/env python3
"""
Basic usage example for CORE-NN.

This example demonstrates:
1. Loading configuration
2. Creating and initializing the model
3. Basic inference and generation
4. Memory operations (remember, recall, forget)
5. Session management
"""

import torch
import sys
from pathlib import Path

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.inference import InferenceEngine, SessionManager
from core_nn.utils import setup_logging, get_optimal_device


def main():
    """Main example function."""
    print("🧠 CORE-NN Basic Usage Example")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Load configuration
    print("📋 Loading configuration...")
    config_manager = ConfigManager()
    
    # Try to load edge device config, fallback to default
    try:
        config = config_manager.load_config("configs/edge_device.yaml")
        print("✅ Loaded edge device configuration")
    except:
        config = config_manager.load_config("configs/default.yaml")
        print("✅ Loaded default configuration")
    
    # Get optimal device
    device = get_optimal_device(config.device.preferred)
    print(f"🖥️  Using device: {device}")
    
    # Create model
    print("\n🏗️  Creating CORE-NN model...")
    model = CoreNNModel(config, vocab_size=1000)  # Small vocab for demo
    model.to(device)
    print("✅ Model created successfully")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Create inference engine
    print("\n⚡ Creating inference engine...")
    inference_engine = InferenceEngine(model, config.inference)
    print("✅ Inference engine ready")
    
    # Create session manager
    print("\n📝 Setting up session management...")
    session_manager = SessionManager(config.session)
    session = session_manager.create_session("Basic Usage Demo")
    print("✅ Session created")
    
    # Start model session
    model.start_session()
    print("🚀 Model session started")
    
    print("\n" + "="*50)
    print("🧪 RUNNING DEMONSTRATIONS")
    print("="*50)
    
    # Demo 1: Basic memory operations
    print("\n1️⃣  Memory Operations Demo")
    print("-" * 30)
    
    # Remember some facts
    facts = [
        "The capital of France is Paris",
        "Python is a programming language", 
        "CORE-NN uses biological memory principles",
        "Edge computing is important for AI"
    ]
    
    for fact in facts:
        result = model.remember(fact)
        print(f"💾 Remembered: {fact}")
        print(f"   Result: {result}")
    
    # Recall information
    queries = ["capital", "programming", "memory", "edge"]
    
    for query in queries:
        memories = model.recall(query, top_k=2)
        print(f"\n🔍 Recalling '{query}':")
        
        episodic = memories.get('episodic_memories', [])
        if episodic:
            for i, memory in enumerate(episodic[:2]):
                print(f"   {i+1}. {memory.instruction}")
        else:
            print("   No memories found")
    
    # Demo 2: Text generation
    print("\n\n2️⃣  Text Generation Demo")
    print("-" * 30)
    
    # Create simple input
    input_text = "Hello world"
    print(f"📝 Input: '{input_text}'")
    
    # Simple tokenization for demo
    input_tokens = [min(ord(c), 999) for c in input_text[:20]]
    input_tokens = input_tokens + [0] * (20 - len(input_tokens))  # Pad
    input_ids = torch.tensor([input_tokens], device=device)
    
    print("🎯 Generating response...")
    
    try:
        # Generate with model
        generation_result = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            temperature=0.8,
            instruction="Generate a friendly response"
        )
        
        print(f"✅ Generated {len(generation_result['generated_tokens'])} tokens")
        print(f"🤖 Generated tokens: {generation_result['generated_tokens']}")
        
        # Add to session
        session.add_interaction(
            user_input=input_text,
            model_response=str(generation_result['generated_tokens']),
            metadata={
                "tokens_generated": len(generation_result['generated_tokens']),
                "response_time": 0.1  # Placeholder
            }
        )
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    # Demo 3: Memory statistics
    print("\n\n3️⃣  Memory Statistics Demo")
    print("-" * 30)
    
    memory_stats = model.get_memory_stats()
    print("📊 Memory Statistics:")
    
    for component, stats in memory_stats.items():
        print(f"   {component}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                print(f"     {key}: {value}")
        else:
            print(f"     {stats}")
    
    # Demo 4: System status
    print("\n\n4️⃣  System Status Demo")
    print("-" * 30)
    
    system_status = model.execution_engine.get_system_status()
    print("🖥️  System Status:")
    
    for category, info in system_status.items():
        print(f"   {category}:")
        if isinstance(info, dict):
            for key, value in info.items():
                print(f"     {key}: {value}")
        else:
            print(f"     {info}")
    
    # Demo 5: Memory optimization
    print("\n\n5️⃣  Memory Optimization Demo")
    print("-" * 30)
    
    print("🧹 Running memory optimization...")
    optimization_result = model.optimize_memory()
    
    print("✅ Optimization completed:")
    for key, value in optimization_result.items():
        print(f"   {key}: {value}")
    
    # Demo 6: Forgetting
    print("\n\n6️⃣  Forgetting Demo")
    print("-" * 30)
    
    forget_query = "programming"
    print(f"🗑️  Forgetting memories about '{forget_query}'...")
    
    forget_result = model.forget(forget_query)
    print("✅ Forgetting completed:")
    for key, value in forget_result.items():
        print(f"   {key}: {value}")
    
    # Verify forgetting worked
    print(f"\n🔍 Checking if '{forget_query}' memories were removed...")
    remaining_memories = model.recall(forget_query, top_k=5)
    episodic_count = len(remaining_memories.get('episodic_memories', []))
    print(f"   Remaining episodic memories: {episodic_count}")
    
    # Demo 7: Session summary
    print("\n\n7️⃣  Session Summary")
    print("-" * 30)
    
    session_summary = session.get_session_summary()
    print("📋 Session Summary:")
    
    for key, value in session_summary.items():
        if key != "stats":
            print(f"   {key}: {value}")
    
    print("   stats:")
    for stat_key, stat_value in session_summary["stats"].items():
        print(f"     {stat_key}: {stat_value}")
    
    # Save session
    print("\n💾 Saving session...")
    session_manager.save_session(session)
    print("✅ Session saved")
    
    # End model session
    model.end_session()
    print("🏁 Model session ended")
    
    print("\n" + "="*50)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    print("\n📚 What you learned:")
    print("• How to load and configure CORE-NN")
    print("• Basic memory operations (remember, recall, forget)")
    print("• Text generation capabilities")
    print("• Memory and system monitoring")
    print("• Session management")
    print("• Memory optimization")
    
    print("\n🚀 Next steps:")
    print("• Try the interactive CLI: python -m core_nn.cli chat")
    print("• Explore different configurations in configs/")
    print("• Check out the test suite in tests/")
    print("• Read the documentation in docs/")


if __name__ == "__main__":
    main()
