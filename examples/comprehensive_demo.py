#!/usr/bin/env python3
"""
Comprehensive CORE-NN API Demonstration

This script demonstrates all major features of the CORE-NN architecture:
- Model initialization and configuration
- Text generation
- Memory operations (remember, recall, forget)
- Knowledge pack operations
- Component-level usage
- Session management

Author: Adrian Paredez (@paredezadrian)
Repository: https://github.com/paredezadrian/core_nn.git
Version: 0.2.2
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add core_nn to path if running from examples directory
sys.path.append(str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.components.rteu import RecursiveTemporalEmbeddingUnit
from core_nn.components.igpm import InstructionGuidedPlasticityModule
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.config.schema import *
from core_nn.memory.kpack import save_kpack, load_kpack, create_capsule
from core_nn.utils import get_optimal_device


def demo_basic_usage():
    """Demonstrate basic model usage."""
    print("🚀 CORE-NN Basic Usage Demo")
    print("=" * 50)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config('configs/default.yaml')
    
    # Adjust for demo (smaller sizes)
    config.rteu.embedding_dim = 256
    config.rteu.num_layers = 2
    config.bcm.memory_size = 64
    config.bcm.embedding_dim = 256
    
    # Create model
    print("Creating CORE-NN model...")
    model = CoreNNModel(config)
    model.eval()
    
    # Generate text
    print("\n📝 Text Generation Demo:")
    input_ids = torch.randint(0, 1000, (1, 8))  # Random input tokens
    
    with torch.no_grad():
        result = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {result['logits'].shape}")
    print(f"Vocab size: {result['logits'].size(-1)}")
    
    return model


def demo_memory_operations(model):
    """Demonstrate memory operations."""
    print("\n🧠 Memory Operations Demo")
    print("=" * 50)
    
    # Start session
    print("Starting session...")
    model.start_session()
    
    # Remember information
    print("\n📚 Remembering information:")
    facts = [
        "The capital of France is Paris",
        "Python is a programming language",
        "CORE-NN uses biological inspiration",
        "Machine learning requires data"
    ]
    
    for fact in facts:
        result = model.remember(fact)
        print(f"  ✓ Remembered: {fact}")
        print(f"    Status: {result['memory_stored']}")
    
    # Recall information
    print("\n🔍 Recalling information:")
    queries = ["capital", "programming", "CORE-NN", "machine learning"]
    
    for query in queries:
        memories = model.recall(query)
        print(f"  Query: '{query}'")
        episodic_memories = memories.get('episodic_memories', [])
        print(f"  Found {len(episodic_memories)} episodic memories")
        for i, memory in enumerate(episodic_memories[:2]):  # Show first 2
            print(f"    {i+1}. {memory.instruction}")
    
    # Memory statistics
    print("\n📊 Memory Statistics:")
    stats = model.get_memory_stats()
    for component, stat in stats.items():
        if isinstance(stat, dict):
            print(f"  {component}:")
            for key, value in stat.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")
    
    # End session
    model.end_session()
    print("\n✅ Session ended")


def demo_component_usage():
    """Demonstrate individual component usage."""
    print("\n🔧 Component Usage Demo")
    print("=" * 50)
    
    device = get_optimal_device()
    
    # BCM Demo
    print("\n🧠 Biological Core Memory (BCM):")
    bcm_config = BCMConfig(memory_size=32, embedding_dim=128, salience_threshold=0.5)
    bcm = BiologicalCoreMemory(bcm_config).to(device)
    
    # Test with different salience inputs
    high_salience = torch.randn(2, 128, device=device) * 2.0  # High variance
    low_salience = torch.randn(2, 128, device=device) * 0.1   # Low variance
    
    output1, info1 = bcm(high_salience)
    print(f"  High salience input - Stored: {info1['num_stored_memories']}, Avg salience: {info1['average_salience']:.3f}")
    
    output2, info2 = bcm(low_salience)
    print(f"  Low salience input - Stored: {info2['num_stored_memories']}, Avg salience: {info2['average_salience']:.3f}")
    
    # RTEU Demo
    print("\n⏰ Recursive Temporal Embedding Unit (RTEU):")
    rteu_config = RTEUConfig(embedding_dim=128, num_layers=2, temporal_scales=[1, 4])
    rteu = RecursiveTemporalEmbeddingUnit(rteu_config).to(device)
    
    # Process temporal sequence
    for t in range(5):
        input_emb = torch.randn(2, 128, device=device)
        output, info = rteu(input_emb)
        if t == 0 or t == 4:
            print(f"  Step {t}: Global state norm: {info['global_state_norm']:.3f}")
    
    # IGPM Demo
    print("\n🧩 Instruction-Guided Plasticity Module (IGPM):")
    igpm_config = IGPMConfig(plastic_slots=16, instruction_embedding_dim=64)
    igpm = InstructionGuidedPlasticityModule(igpm_config, vocab_size=1000, embedding_dim=128).to(device)
    
    input_emb = torch.randn(2, 128, device=device)
    instructions = ["remember this", "focus on details", "ignore noise"]
    
    for instruction in instructions:
        output, info = igpm(input_emb, instruction=instruction)
        change = torch.norm(output - input_emb, dim=-1).mean().item()
        print(f"  '{instruction}': Change magnitude: {change:.4f}, Active slots: {len(info['relevant_slots'])}")
    
    # MLCS Demo
    print("\n📦 Multi-Level Compression Synthesizer (MLCS):")
    mlcs_config = MLCSConfig(latent_dim=64, num_compression_levels=3)
    mlcs = MultiLevelCompressionSynthesizer(mlcs_config, input_dim=128).to(device)
    
    knowledge = torch.randn(4, 128, device=device)
    kpack = mlcs.compress_knowledge(knowledge, name="demo_knowledge")
    
    print(f"  Original size: {knowledge.numel()} parameters")
    print(f"  Compressed size: {kpack.compressed_size} parameters")
    print(f"  Compression ratio: {kpack.compressed_size / kpack.original_size:.4f}")
    
    # Test decompression
    reconstructed = mlcs.decompress_knowledge(kpack)
    error = torch.nn.functional.mse_loss(reconstructed, knowledge).item()
    print(f"  Reconstruction error: {error:.6f}")


def demo_knowledge_packs():
    """Demonstrate knowledge pack operations."""
    print("\n📦 Knowledge Pack Demo")
    print("=" * 50)
    
    # Create knowledge pack
    print("Creating knowledge pack...")
    
    # Create some sample knowledge
    embeddings = [torch.randn(128) for _ in range(10)]
    
    # Create capsule
    capsule = create_capsule(
        name="demo_knowledge",
        topic="demonstration",
        description="Demonstration knowledge pack",
        embeddings={"demo_embeddings": torch.stack(embeddings)},
        creator="demo",
        version="1.0"
    )
    
    print(f"Created capsule: {capsule.metadata.name}")
    print(f"Size: {capsule.calculate_size()} bytes")
    print(f"Embeddings: {len(capsule.embeddings)}")
    
    # Save to file
    kpack_path = "demo_knowledge.kpack"
    save_kpack(capsule, kpack_path)
    print(f"Saved to: {kpack_path}")
    
    # Load from file
    loaded_capsule = load_kpack(kpack_path)
    print(f"Loaded: {loaded_capsule.metadata.name}")
    print(f"Description: {loaded_capsule.metadata.description}")
    
    # Clean up
    Path(kpack_path).unlink(missing_ok=True)
    print("Cleaned up demo file")


def demo_custom_configuration():
    """Demonstrate custom configuration."""
    print("\n⚙️ Custom Configuration Demo")
    print("=" * 50)
    
    # Create custom configuration
    custom_config = CoreNNConfig(
        model=ModelConfig(
            name="demo-model",
            version="0.2.2"
        ),
        bcm=BCMConfig(
            memory_size=128,
            embedding_dim=256,
            salience_threshold=0.6,
            decay_rate=0.9
        ),
        rteu=RTEUConfig(
            num_layers=3,
            embedding_dim=256,
            temporal_scales=[1, 2, 4, 8],
            num_capsules=8
        ),
        igpm=IGPMConfig(
            plastic_slots=32,
            meta_learning_rate=0.005,
            instruction_embedding_dim=128
        ),
        mlcs=MLCSConfig(
            latent_dim=128,
            num_compression_levels=4,
            compression_ratio=0.05
        ),
        inference=InferenceConfig(
            max_sequence_length=1024,
            temperature=0.8,
            top_k=40,
            top_p=0.95
        )
    )
    
    print("Custom configuration created:")
    print(f"  Model: {custom_config.model.name} v{custom_config.model.version}")
    print(f"  BCM memory size: {custom_config.bcm.memory_size}")
    print(f"  RTEU layers: {custom_config.rteu.num_layers}")
    print(f"  IGPM plastic slots: {custom_config.igpm.plastic_slots}")
    print(f"  MLCS compression levels: {custom_config.mlcs.num_compression_levels}")
    
    # Create model with custom config
    print("\nCreating model with custom configuration...")
    model = CoreNNModel(custom_config)
    print("✅ Custom model created successfully")
    
    return model


def main():
    """Run comprehensive demo."""
    print("🎯 CORE-NN Comprehensive API Demo")
    print("=" * 60)
    
    try:
        # Basic usage
        model = demo_basic_usage()
        
        # Memory operations
        demo_memory_operations(model)
        
        # Component usage
        demo_component_usage()
        
        # Knowledge packs
        demo_knowledge_packs()
        
        # Custom configuration
        custom_model = demo_custom_configuration()
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("All CORE-NN features demonstrated:")
        print("✅ Basic model usage")
        print("✅ Memory operations (remember/recall/forget)")
        print("✅ Individual component usage")
        print("✅ Knowledge pack operations")
        print("✅ Custom configuration")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
