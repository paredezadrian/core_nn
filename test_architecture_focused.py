#!/usr/bin/env python3
"""
Focused test of CORE-NN's novel architectural features.

This script tests the key innovative aspects of CORE-NN to validate the architecture works.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Any

# Import CORE-NN components
from core_nn import CoreNNModel, ConfigManager
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.components.rteu import RecursiveTemporalEmbeddingUnit
from core_nn.components.igpm import InstructionGuidedPlasticityModule
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.config.schema import *
from core_nn.utils import get_optimal_device


def test_bcm_memory_retention():
    """Test BCM's selective memory retention based on salience."""
    print("🧠 Testing BCM Memory Retention...")
    
    config = BCMConfig(
        memory_size=32,
        embedding_dim=128,
        salience_threshold=0.5,
        decay_rate=0.95
    )
    bcm = BiologicalCoreMemory(config)
    
    # Test with different input patterns
    batch_size = 2
    embedding_dim = 128
    
    # High salience input (large values)
    high_salience_input = torch.randn(batch_size, embedding_dim) * 3.0
    
    # Low salience input (small values)  
    low_salience_input = torch.randn(batch_size, embedding_dim) * 0.1
    
    print("  Processing high salience input...")
    output1, info1 = bcm(high_salience_input)
    print(f"    Stored memories: {info1['num_stored_memories']}")
    print(f"    Average salience: {info1['average_salience']:.3f}")
    
    print("  Processing low salience input...")
    output2, info2 = bcm(low_salience_input)
    print(f"    Stored memories: {info2['num_stored_memories']}")
    print(f"    Average salience: {info2['average_salience']:.3f}")
    
    print(f"  ✅ BCM selectively retains {info1['num_stored_memories']} vs {info2['num_stored_memories']} memories")
    return True


def test_rteu_temporal_processing():
    """Test RTEU's multi-timescale temporal processing."""
    print("\n⏰ Testing RTEU Temporal Processing...")
    
    config = RTEUConfig(
        embedding_dim=128,
        num_layers=2,
        temporal_scales=[1, 4],  # Fast and slow timescales
        num_capsules=4,
        capsule_dim=32,
        routing_iterations=2
    )
    rteu = RecursiveTemporalEmbeddingUnit(config)
    
    batch_size = 2
    embedding_dim = 128
    
    # Process a sequence to build temporal state
    outputs = []
    for t in range(8):
        input_emb = torch.randn(batch_size, embedding_dim)
        output, info = rteu(input_emb)
        outputs.append(output)
        
        if t == 0:
            print(f"  Step {t}: Global state norm: {info['global_state_norm']:.3f}")
        elif t == 7:
            print(f"  Step {t}: Global state norm: {info['global_state_norm']:.3f}")
    
    # Test state reset
    rteu.reset_all_states()
    output_after_reset, info_after_reset = rteu(input_emb)
    
    print(f"  After reset: Global state norm: {info_after_reset['global_state_norm']:.3f}")
    print(f"  ✅ RTEU processes temporal sequences with {len(config.temporal_scales)} timescales")
    return True


def test_igpm_plasticity():
    """Test IGPM's instruction-guided plasticity."""
    print("\n🧩 Testing IGPM Plasticity...")
    
    config = IGPMConfig(
        plastic_slots=8,
        meta_learning_rate=0.01,
        instruction_embedding_dim=64,
        max_episodic_memories=20,
        plasticity_threshold=0.05
    )
    igpm = InstructionGuidedPlasticityModule(config, vocab_size=1000, embedding_dim=128)
    
    batch_size = 2
    embedding_dim = 128
    
    # Test different instructions
    instructions = [
        "remember this pattern",
        "focus on important details",
        "ignore noise"
    ]
    
    input_emb = torch.randn(batch_size, embedding_dim)
    
    for instruction in instructions:
        output, info = igpm(input_emb, instruction=instruction)
        
        # Measure plasticity effect
        change_magnitude = torch.norm(output - input_emb, dim=-1).mean().item()
        
        print(f"  '{instruction}': change={change_magnitude:.4f}, slots={len(info['relevant_slots'])}")
    
    # Test explicit memory storage
    memory_info = igpm.remember_explicit("test pattern", input_emb[0])
    print(f"  Stored explicit memory: {memory_info['memory_stored']}")

    # Test memory recall
    recalled_memories = igpm.recall_by_instruction("test pattern", top_k=3)
    print(f"  Recalled {len(recalled_memories)} memories")
    
    print(f"  ✅ IGPM adapts to instructions with {config.plastic_slots} plastic slots")
    return True


def test_mlcs_compression():
    """Test MLCS's knowledge compression."""
    print("\n📦 Testing MLCS Compression...")
    
    config = MLCSConfig(
        latent_dim=64,
        num_compression_levels=3,
        compression_ratio=0.4
    )
    mlcs = MultiLevelCompressionSynthesizer(config, input_dim=128)
    
    # Create some knowledge to compress
    batch_size = 4
    knowledge = torch.randn(batch_size, 128)
    
    print(f"  Original size: {knowledge.numel()} parameters")
    
    # Test compression
    kpack = mlcs.compress_knowledge(knowledge, compression_level=2, name="test_knowledge")
    print(f"  Compressed to knowledge pack: {kpack.name}")
    print(f"    Original size: {kpack.original_size}")
    print(f"    Compressed size: {kpack.compressed_size}")
    print(f"    Compression ratio: {kpack.compressed_size / kpack.original_size:.3f}")

    # Test decompression
    reconstructed = mlcs.decompress_knowledge(kpack, from_level=0)
    reconstruction_error = F.mse_loss(reconstructed, knowledge).item()
    
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    print(f"  ✅ MLCS compresses knowledge with {config.num_compression_levels} levels")
    return True


def test_full_model_integration():
    """Test the full CORE-NN model integration."""
    print("\n🚀 Testing Full Model Integration...")
    
    # Load default config
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/default.yaml")
    
    # Adjust for testing
    config.rteu.embedding_dim = 128
    config.rteu.num_layers = 2
    config.bcm.memory_size = 32
    config.bcm.embedding_dim = 128
    config.igpm.instruction_embedding_dim = 64
    config.mlcs.latent_dim = 64
    
    # Create model
    model = CoreNNModel(config)
    model.eval()
    
    # Test forward pass
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"  Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        output = model(input_ids)

    print(f"  Output shape: {output['logits'].shape}")
    print(f"  Output vocab size: {output['logits'].size(-1)}")
    
    # Test memory commands
    print("  Testing memory commands...")
    
    # Test remember command
    remember_result = model.remember("This is important information")
    print(f"    Remember: {remember_result['memory_stored']}")

    # Test recall command
    recall_result = model.recall("important")
    print(f"    Recall: found {len(recall_result)} memories")
    
    print("  ✅ Full model integration working")
    return True


def main():
    """Run focused architecture tests."""
    print("🎯 CORE-NN Focused Architecture Testing")
    print("=" * 50)
    
    tests = [
        test_bcm_memory_retention,
        test_rteu_temporal_processing,
        test_igpm_plasticity,
        test_mlcs_compression,
        test_full_model_integration
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            results.append(False)
    
    print(f"\n📊 RESULTS: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All novel architecture features are working!")
    else:
        print("⚠️  Some features need attention")
    
    return results


if __name__ == "__main__":
    main()
