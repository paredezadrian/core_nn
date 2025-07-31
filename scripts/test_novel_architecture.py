#!/usr/bin/env python3
"""
Comprehensive test of CORE-NN's novel architectural features.

This script tests the innovative aspects that differentiate CORE-NN from traditional transformers:
1. BCM's salience-based memory retention vs full attention
2. RTEU's multi-timescale temporal processing 
3. IGPM's instruction-guided plasticity without global updates
4. MLCS's dynamic knowledge compression and loading
5. Edge-efficient modular execution patterns
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Any

# Import CORE-NN components
from core_nn import CoreNNModel, ConfigManager
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.components.rteu import RecursiveTemporalEmbeddingUnit
from core_nn.components.igpm import InstructionGuidedPlasticityModule
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.config.schema import *
from core_nn.utils import get_optimal_device


class ArchitectureNoveltyTester:
    """Test suite for CORE-NN's novel architectural features."""
    
    def __init__(self):
        self.device = get_optimal_device()
        self.results = {}
        
    def test_bcm_salience_vs_attention(self):
        """Test BCM's salience-based memory vs traditional full attention."""
        print("\n🧠 Testing BCM Salience-Based Memory...")
        
        # Create BCM with different salience thresholds
        config = BCMConfig(
            memory_size=64,
            embedding_dim=256,
            salience_threshold=0.3,
            decay_rate=0.01
        )
        bcm = BiologicalCoreMemory(config).to(self.device)
        
        # Generate sequence with varying importance
        batch_size = 4
        seq_len = 20
        embedding_dim = 256
        
        # Create inputs with different salience levels
        important_embeddings = torch.randn(batch_size, embedding_dim, device=self.device) * 2.0  # High variance = high salience
        normal_embeddings = torch.randn(batch_size, embedding_dim, device=self.device) * 0.5    # Low variance = low salience
        
        memory_stats = []
        
        # Process sequence
        for i in range(seq_len):
            if i % 5 == 0:  # Every 5th input is "important"
                input_emb = important_embeddings
            else:
                input_emb = normal_embeddings
                
            output, info = bcm(input_emb)
            memory_stats.append({
                'step': i,
                'memory_usage': info['num_stored_memories'],
                'avg_salience': info['average_salience'],
                'memory_utilization': info['memory_utilization']
            })
        
        # Analyze results
        important_steps = [s for s in memory_stats if s['step'] % 5 == 0]
        normal_steps = [s for s in memory_stats if s['step'] % 5 != 0]
        
        avg_important_salience = np.mean([s['avg_salience'] for s in important_steps])
        avg_normal_salience = np.mean([s['avg_salience'] for s in normal_steps])
        
        print(f"  📊 Average salience for important inputs: {avg_important_salience:.3f}")
        print(f"  📊 Average salience for normal inputs: {avg_normal_salience:.3f}")
        print(f"  📊 Salience ratio: {avg_important_salience/avg_normal_salience:.2f}x")
        print(f"  📊 Final memory usage: {memory_stats[-1]['memory_usage']}/{config.memory_size}")

        self.results['bcm_salience'] = {
            'important_salience': avg_important_salience,
            'normal_salience': avg_normal_salience,
            'salience_ratio': avg_important_salience/avg_normal_salience,
            'memory_efficiency': memory_stats[-1]['memory_usage'] / config.memory_size
        }
        
        return memory_stats
    
    def test_rteu_multitimescale_processing(self):
        """Test RTEU's multi-timescale temporal processing capabilities."""
        print("\n⏰ Testing RTEU Multi-Timescale Processing...")
        
        config = RTEUConfig(
            embedding_dim=256,
            num_layers=4,
            temporal_scales=[1, 2, 4, 8],  # Different timescales
            num_capsules=8,
            capsule_dim=64,
            routing_iterations=3
        )
        rteu = RecursiveTemporalEmbeddingUnit(config).to(self.device)
        
        # Create temporal patterns at different scales
        batch_size = 2
        seq_len = 32
        embedding_dim = 256
        
        # Pattern 1: Fast oscillation (period=2)
        # Pattern 2: Medium oscillation (period=4) 
        # Pattern 3: Slow oscillation (period=8)
        
        temporal_outputs = []
        
        for t in range(seq_len):
            # Create input with multiple temporal patterns
            fast_pattern = torch.sin(torch.tensor(t * 2 * np.pi / 2))
            medium_pattern = torch.sin(torch.tensor(t * 2 * np.pi / 4))
            slow_pattern = torch.sin(torch.tensor(t * 2 * np.pi / 8))
            
            # Embed patterns in different dimensions
            input_emb = torch.randn(batch_size, embedding_dim, device=self.device)
            input_emb[:, :64] *= fast_pattern
            input_emb[:, 64:128] *= medium_pattern  
            input_emb[:, 128:192] *= slow_pattern
            
            output, info = rteu(input_emb)
            temporal_outputs.append({
                'step': t,
                'output': output.detach(),
                'layer_activations': info['layer_activations'],
                'global_state_norm': info['global_state_norm']
            })
        
        # Analyze temporal patterns captured
        print(f"  📊 Processed {seq_len} timesteps with {len(config.temporal_scales)} timescales")
        print(f"  📊 Final global state norm: {temporal_outputs[-1]['global_state_norm']:.3f}")
        print(f"  📊 Output stability: {torch.std(torch.stack([t['output'] for t in temporal_outputs[-5:]])).item():.3f}")
        
        self.results['rteu_temporal'] = {
            'timescales': config.temporal_scales,
            'final_state_norm': temporal_outputs[-1]['global_state_norm'],
            'output_stability': torch.std(torch.stack([t['output'] for t in temporal_outputs[-5:]])).item()
        }
        
        return temporal_outputs
    
    def test_igpm_instruction_plasticity(self):
        """Test IGPM's instruction-guided plasticity without global updates."""
        print("\n🧩 Testing IGPM Instruction-Guided Plasticity...")
        
        config = IGPMConfig(
            plastic_slots=16,
            meta_learning_rate=0.01,
            instruction_embedding_dim=128,
            max_episodic_memories=50,
            plasticity_threshold=0.1
        )
        igpm = InstructionGuidedPlasticityModule(config, vocab_size=1000, embedding_dim=256).to(self.device)
        
        # Test different instructions and their effects
        instructions = [
            "remember this is important",
            "focus on the key details", 
            "ignore irrelevant information",
            "connect this to previous knowledge"
        ]
        
        batch_size = 2
        embedding_dim = 256
        
        plasticity_results = []
        
        for i, instruction in enumerate(instructions):
            input_emb = torch.randn(batch_size, embedding_dim, device=self.device)
            
            # Test plasticity response
            output, info = igpm(input_emb, instruction=instruction)
            
            # Measure plasticity effect
            plasticity_magnitude = torch.norm(output - input_emb, dim=-1).mean().item()
            
            plasticity_results.append({
                'instruction': instruction,
                'plasticity_magnitude': plasticity_magnitude,
                'relevant_slots': info['relevant_slots'],
                'slot_activations': info.get('slot_activations', [])
            })
            
            print(f"  📝 '{instruction}': plasticity={plasticity_magnitude:.3f}, slots={len(info['relevant_slots'])}")
        
        # Test memory recall
        print("  🔍 Testing memory recall...")
        recall_output, recall_info = igpm.recall_memory("remember this is important")
        
        self.results['igpm_plasticity'] = {
            'instructions_tested': len(instructions),
            'avg_plasticity': np.mean([r['plasticity_magnitude'] for r in plasticity_results]),
            'memory_recall_success': recall_output is not None
        }
        
        return plasticity_results
    
    def test_mlcs_knowledge_compression(self):
        """Test MLCS's knowledge compression and dynamic loading."""
        print("\n📦 Testing MLCS Knowledge Compression...")
        
        config = MLCSConfig(
            latent_dim=128,
            num_compression_levels=4,
            compression_ratios=[0.8, 0.6, 0.4, 0.2]
        )
        mlcs = MultiLevelCompressionSynthesizer(config, embedding_dim=256).to(self.device)
        
        # Create knowledge to compress
        batch_size = 8
        knowledge_dim = 256
        knowledge_data = torch.randn(batch_size, knowledge_dim, device=self.device)
        
        # Test compression at different levels
        compression_results = []
        
        for level in range(1, config.num_compression_levels + 1):
            start_time = time.time()
            
            # Compress
            compressed = mlcs.compress(knowledge_data, target_level=level)
            
            # Decompress
            reconstructed = mlcs.decompress(compressed, from_level=level-1)
            
            compression_time = time.time() - start_time
            
            # Measure compression quality
            reconstruction_error = F.mse_loss(reconstructed, knowledge_data).item()
            compression_ratio = compressed[-1].numel() / knowledge_data.numel()
            
            compression_results.append({
                'level': level,
                'compression_ratio': compression_ratio,
                'reconstruction_error': reconstruction_error,
                'compression_time': compression_time
            })
            
            print(f"  📊 Level {level}: ratio={compression_ratio:.3f}, error={reconstruction_error:.4f}, time={compression_time:.3f}s")
        
        self.results['mlcs_compression'] = {
            'levels_tested': config.num_compression_levels,
            'best_compression_ratio': min([r['compression_ratio'] for r in compression_results]),
            'avg_reconstruction_error': np.mean([r['reconstruction_error'] for r in compression_results])
        }
        
        return compression_results


def main():
    """Run comprehensive architecture novelty tests."""
    print("🚀 CORE-NN Novel Architecture Testing")
    print("=" * 50)
    
    tester = ArchitectureNoveltyTester()
    
    # Test each novel component
    bcm_results = tester.test_bcm_salience_vs_attention()
    rteu_results = tester.test_rteu_multitimescale_processing()
    igpm_results = tester.test_igpm_instruction_plasticity()
    mlcs_results = tester.test_mlcs_knowledge_compression()
    
    # Summary
    print("\n📋 ARCHITECTURE NOVELTY SUMMARY")
    print("=" * 50)
    
    for component, results in tester.results.items():
        print(f"\n{component.upper()}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n✅ Novel architecture testing complete!")
    return tester.results


if __name__ == "__main__":
    main()
