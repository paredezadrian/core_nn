#!/usr/bin/env python3
"""
Test script for IGMP plasticity improvements.
Tests the enhanced gradient-based plasticity mechanisms.
"""

import torch
import torch.nn.functional as F
from core_nn.config.schema import IGPMConfig
from core_nn.components.igpm import InstructionGuidedPlasticityModule

def test_enhanced_plasticity():
    """Test enhanced IGPM plasticity with gradient-based updates."""
    print("🧩 Testing Enhanced IGPM Plasticity...")
    
    # Configuration with lower threshold for easier activation
    config = IGPMConfig(
        plastic_slots=4,
        meta_learning_rate=0.1,  # Higher learning rate
        instruction_embedding_dim=64,
        max_episodic_memories=20,
        plasticity_threshold=0.01  # Much lower threshold
    )
    
    igmp = InstructionGuidedPlasticityModule(config, vocab_size=1000, embedding_dim=128)
    
    batch_size = 2
    embedding_dim = 128
    
    # Test instructions
    instructions = [
        "amplify this signal",
        "remember this pattern", 
        "focus on important features",
        "suppress noise"
    ]
    
    print("\n📊 Testing plasticity responses:")
    
    for i, instruction in enumerate(instructions):
        # Create input with some structure
        input_emb = torch.randn(batch_size, embedding_dim) * 0.5
        
        # Test initial response
        output_before, info_before = igmp(input_emb, instruction=instruction)
        change_before = torch.norm(output_before - input_emb, dim=-1).mean().item()
        
        # Create a target that's different from input to encourage learning
        target = input_emb + torch.randn_like(input_emb) * 0.2
        
        # Learn from instruction using both methods
        learn_info = igmp.learn_from_instruction(instruction, input_emb, target)

        # Test MAML learning if available
        try:
            maml_info = igmp.maml_learn_from_instruction(instruction, input_emb, target)
            maml_available = True
        except:
            maml_available = False
            maml_info = {}

        # Test response after learning
        output_after, info_after = igmp(input_emb, instruction=instruction)
        change_after = torch.norm(output_after - input_emb, dim=-1).mean().item()

        print(f"  '{instruction}':")
        print(f"    Before learning: change={change_before:.4f}")
        print(f"    After learning:  change={change_after:.4f}")
        print(f"    Learning loss:   {learn_info['loss']:.4f}")
        print(f"    Weight update:   {learn_info['weight_update_norm']:.4f}")
        print(f"    Plasticity str:  {learn_info.get('plasticity_strength', 'N/A')}")
        print(f"    Active slots:    {len(info_after['relevant_slots'])}")
        print(f"    Total effect:    {info_after.get('total_plasticity_effect', 'N/A')}")
        if maml_available:
            print(f"    MAML available:  ✅")

        # Test neuromodulation if available
        try:
            # Get a slot to test neuromodulation
            slot = igmp.slots[0]
            if hasattr(slot, 'dopamine_level'):
                print(f"    Neuromod levels: DA={slot.dopamine_level.item():.2f}, ACh={slot.acetylcholine_level.item():.2f}")
                print(f"                     NE={slot.norepinephrine_level.item():.2f}, 5HT={slot.serotonin_level.item():.2f}")

                # Test neuromodulation update
                slot.update_neuromodulation(reward_signal=0.8, attention_demand=0.9)
                neuromod_factor = slot._compute_neuromodulation_factor()
                print(f"    Neuromod factor: {neuromod_factor:.3f}")
        except:
            pass

        # Check if plasticity improved
        improvement = change_after - change_before
        if improvement > 0.001:
            print(f"    ✅ Plasticity improved by {improvement:.4f}")
        elif change_after > 1.0:  # Check if there's significant plasticity at all
            print(f"    ✅ Strong plasticity response: {change_after:.4f}")
        else:
            print(f"    ❌ No significant plasticity improvement")
        print()
    
    print("🔍 Testing slot usage patterns:")
    for i, slot in enumerate(igmp.slots):
        usage = igmp.slot_usage[i].item()
        plasticity_str = slot.plasticity_strength.item()
        print(f"  Slot {i}: usage={usage:.1f}, plasticity_strength={plasticity_str:.3f}")
    
    return True

def test_gradient_flow():
    """Test that gradients flow properly through the enhanced plasticity mechanism."""
    print("\n🔬 Testing Gradient Flow...")
    
    config = IGPMConfig(
        plastic_slots=2,
        meta_learning_rate=0.05,
        instruction_embedding_dim=32,
        plasticity_threshold=0.01
    )
    
    igmp = InstructionGuidedPlasticityModule(config, vocab_size=100, embedding_dim=64)
    
    # Create simple test case
    input_tensor = torch.randn(1, 64, requires_grad=True)
    target = torch.randn(1, 64)
    instruction = "test gradient flow"
    
    # Forward pass
    output, _ = igmp(input_tensor, instruction=instruction)
    
    # Compute loss
    loss = F.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    if input_tensor.grad is not None:
        grad_norm = torch.norm(input_tensor.grad).item()
        print(f"  Input gradient norm: {grad_norm:.4f}")
        if grad_norm > 1e-6:
            print("  ✅ Gradients flowing properly")
        else:
            print("  ❌ Gradients too small")
    else:
        print("  ❌ No gradients computed")
    
    return True

if __name__ == "__main__":
    print("🚀 Testing IGPM Plasticity Improvements")
    print("=" * 50)
    
    try:
        test_enhanced_plasticity()
        test_gradient_flow()
        print("🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
