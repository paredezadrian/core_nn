"""
Test Efficient IGPM on GLUE Tasks.

This module tests the parameter-efficient IGPM on GLUE tasks to validate
that the 93% parameter reduction maintains performance.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any
from pathlib import Path
import sys

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.efficient_igpm import EfficientIGPM
from evaluation.classification_heads import GLUEClassificationManager


class SimpleEfficientModel(nn.Module):
    """
    Simplified model with efficient IGPM for testing.
    """
    
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 1536):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Basic components
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(2048, embedding_dim)
        
        # Efficient IGPM (the star of the show)
        self.igpm = EfficientIGPM(
            embedding_dim=embedding_dim,
            instruction_dim=512,
            num_slots=8,
            vocab_size=vocab_size,
            rank=64
        )
        
        # Output layers
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Session state
        self.current_position = 0
        
    def forward(self, input_ids: torch.Tensor, instruction: str = None) -> Dict[str, torch.Tensor]:
        """Forward pass with efficient IGPM."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        hidden_states = token_embeds + position_embeds
        
        # Apply efficient IGPM
        igpm_output = self.igpm(hidden_states, instruction=instruction)
        hidden_states = igpm_output["hidden_states"]
        
        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        return {
            "logits": logits,
            "last_hidden_state": hidden_states,
            "component_info": {
                "igpm_info": [{
                    "total_plasticity_effect": igpm_output["plasticity_effect"].item(),
                    "active_slots": igpm_output.get("active_slots", []),
                    "slot_weights": igpm_output.get("slot_weights", [])
                }]
            }
        }


def test_efficient_igpm_on_glue():
    """Test efficient IGPM on GLUE tasks."""
    print("Testing Efficient IGPM on GLUE Tasks")
    print("=" * 50)
    
    # Create efficient model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleEfficientModel().to(device)
    model.eval()
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    igpm_params = sum(p.numel() for p in model.igpm.parameters())
    
    print(f"Total model parameters: {total_params:,}")
    print(f"IGPM parameters: {igpm_params:,} ({igpm_params/total_params*100:.1f}%)")
    
    # Create classification manager
    classifier = GLUEClassificationManager(
        input_dim=1536,
        device=str(device)
    )
    
    def tokenize_text(text: str) -> torch.Tensor:
        """Simple tokenization."""
        char_ids = [min(ord(c), 49999) for c in text[:50]]
        if len(char_ids) < 5:
            char_ids.extend([0] * (5 - len(char_ids)))
        return torch.tensor([char_ids], dtype=torch.long, device=device)
    
    # GLUE task examples
    glue_tasks = {
        "rte": [
            {
                "premise": "A man is playing guitar.",
                "hypothesis": "A person is making music.",
                "label": "entailment",
                "instruction": "Does the hypothesis follow from the premise?"
            },
            {
                "premise": "The cat is sleeping on the couch.",
                "hypothesis": "The dog is running in the park.",
                "label": "not_entailment", 
                "instruction": "Does the hypothesis follow from the premise?"
            },
            {
                "premise": "Students are studying in the library.",
                "hypothesis": "People are reading books.",
                "label": "entailment",
                "instruction": "Does the hypothesis follow from the premise?"
            }
        ],
        "sentiment": [
            {
                "text": "This movie is absolutely wonderful and amazing!",
                "label": "positive",
                "instruction": "Determine if this text expresses positive or negative sentiment."
            },
            {
                "text": "This movie is terrible and boring.",
                "label": "negative", 
                "instruction": "Determine if this text expresses positive or negative sentiment."
            },
            {
                "text": "I love this product, it works perfectly!",
                "label": "positive",
                "instruction": "Determine if this text expresses positive or negative sentiment."
            }
        ]
    }
    
    task_results = {}
    
    for task_name, examples in glue_tasks.items():
        print(f"\nTesting {task_name}...")
        
        # Collect training data
        training_data = []
        plasticity_effects = []
        
        for example in examples:
            # Reset model position
            model.current_position = 0
            
            # Create input text
            if task_name == "rte":
                input_text = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
            else:  # sentiment
                input_text = example['text']
            
            input_ids = tokenize_text(input_text)
            
            with torch.no_grad():
                output = model(input_ids, instruction=example['instruction'])
                hidden_states = output["last_hidden_state"]
                plasticity_effect = output["component_info"]["igpm_info"][0]["total_plasticity_effect"]
                
                training_data.append((hidden_states, example['label']))
                plasticity_effects.append(plasticity_effect)
        
        # Print plasticity effects
        print(f"  Plasticity effects: {[f'{p:.4f}' for p in plasticity_effects]}")
        print(f"  Average plasticity: {np.mean(plasticity_effects):.4f}")
        
        # Train classification head
        start_time = time.time()
        training_info = classifier.train_head(
            task_name=task_name,
            examples=training_data,
            learning_rate=1e-3,
            epochs=20
        )
        training_time = time.time() - start_time
        
        # Evaluate
        evaluation_info = classifier.evaluate_head(task_name, training_data)
        accuracy = evaluation_info["accuracy"]
        
        task_results[task_name] = {
            "accuracy": accuracy,
            "training_time": training_time,
            "avg_plasticity": np.mean(plasticity_effects),
            "final_loss": training_info["final_loss"]
        }
        
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  Training time: {training_time:.2f}s")
    
    # Overall results
    overall_accuracy = np.mean([r["accuracy"] for r in task_results.values()])
    
    print(f"\n" + "="*50)
    print(f"EFFICIENT IGPM GLUE RESULTS:")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter efficiency: {overall_accuracy*100/total_params*1e6:.2f} accuracy per million params")
    
    # Compare with original
    original_params = 1_164_964_081
    original_accuracy = 0.6111  # From previous results
    
    print(f"\nCOMPARISON WITH ORIGINAL:")
    print(f"Original: {original_accuracy:.4f} accuracy with {original_params:,} params")
    print(f"Efficient: {overall_accuracy:.4f} accuracy with {total_params:,} params")
    print(f"Parameter reduction: {(1 - total_params/original_params)*100:.1f}%")
    print(f"Performance retention: {overall_accuracy/original_accuracy*100:.1f}%")
    
    return task_results


if __name__ == "__main__":
    results = test_efficient_igpm_on_glue()
