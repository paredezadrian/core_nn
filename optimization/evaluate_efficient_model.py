"""
Evaluate Efficient CORE-NN Model using the standard evaluation framework.

This module runs the efficient CORE-NN through the same evaluation pipeline
to compare performance with the original model.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.efficient_igpm import EfficientIGPM
from evaluation.classification_heads import GLUEClassificationManager
from evaluation.evaluation_framework import EvaluationConfig, EvaluationResult


class EfficientModelEvaluator:
    """Evaluates efficient CORE-NN using the standard evaluation framework."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create simplified efficient model for evaluation
        self.model = self._create_efficient_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Classification manager
        self.classification_manager = GLUEClassificationManager(
            input_dim=1536,
            device=str(self.device)
        )
        
        print(f"Loaded Efficient CORE-NN model on {self.device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    def _create_efficient_model(self):
        """Create simplified efficient model for evaluation."""
        class SimpleEfficientModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding_dim = 1536
                
                # Basic components
                self.token_embedding = torch.nn.Embedding(50000, self.embedding_dim)
                self.position_embedding = torch.nn.Embedding(2048, self.embedding_dim)
                
                # Efficient IGPM
                self.igpm = EfficientIGPM(
                    embedding_dim=self.embedding_dim,
                    instruction_dim=512,
                    num_slots=8,
                    vocab_size=50000,
                    rank=64
                )
                
                # Output layers
                self.norm = torch.nn.LayerNorm(self.embedding_dim)
                self.output_projection = torch.nn.Linear(self.embedding_dim, 50000)
                
                # Session state
                self.current_position = 0
                
            def forward(self, input_ids: torch.Tensor, instruction: str = None) -> Dict[str, torch.Tensor]:
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
                
                # Embeddings
                token_embeds = self.token_embedding(input_ids)
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
        
        return SimpleEfficientModel()
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple tokenization for evaluation."""
        max_len = 50
        char_ids = [min(ord(c), 49999) for c in text[:max_len]]
        if len(char_ids) < 5:
            char_ids.extend([0] * (5 - len(char_ids)))
        return torch.tensor([char_ids], dtype=torch.long, device=self.device)
    
    def _measure_memory_usage(self) -> float:
        """Measure memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return 0.0  # Simplified for CPU
    
    def evaluate_plasticity(self) -> EvaluationResult:
        """Evaluate IGPM plasticity using the same methodology as the original framework."""
        print("Evaluating Efficient IGPM Plasticity...")
        
        start_time = time.time()
        start_memory = self._measure_memory_usage()
        
        # Test instructions (same as original framework)
        instructions = [
            "remember this important information",
            "focus your attention on key details",
            "suppress irrelevant noise",
            "amplify the signal strength",
            "adapt to this new pattern",
            "learn from this example",
            "ignore distracting elements",
            "enhance important features"
        ]
        
        plasticity_magnitudes = []
        adaptation_speeds = []
        context_sensitivities = []
        
        for instruction in instructions:
            # Test plasticity response
            test_input = "Test input for plasticity measurement."
            input_ids = self._tokenize_text(test_input)
            
            with torch.no_grad():
                # Reset model position
                self.model.current_position = 0
                
                # Baseline response
                baseline_output = self.model.forward(input_ids)
                baseline_embedding = baseline_output["last_hidden_state"]
                
                # Reset position again
                self.model.current_position = 0
                
                # Response with instruction
                instruction_output = self.model.forward(input_ids, instruction=instruction)
                instruction_embedding = instruction_output["last_hidden_state"]
                
                # Calculate plasticity magnitude
                change_magnitude = torch.norm(instruction_embedding - baseline_embedding, dim=-1).mean().item()
                plasticity_magnitudes.append(change_magnitude)
                
                # Simplified adaptation speed (using plasticity effect)
                plasticity_effect = instruction_output["component_info"]["igpm_info"][0]["total_plasticity_effect"]
                adaptation_speeds.append(plasticity_effect)
                
                # Context sensitivity (variance across instructions)
                context_sensitivities.append(plasticity_effect)
        
        # Calculate metrics
        avg_plasticity = float(np.mean(plasticity_magnitudes))
        avg_adaptation = float(np.mean(adaptation_speeds))
        avg_context = float(np.std(context_sensitivities)) if len(context_sensitivities) > 1 else 0.0
        
        # Overall score (weighted combination)
        overall_score = float(avg_plasticity * 0.4 + avg_adaptation * 0.3 + avg_context * 0.3)
        
        execution_time = time.time() - start_time
        memory_usage = self._measure_memory_usage() - start_memory
        
        metrics = {
            "average_plasticity_magnitude": avg_plasticity,
            "average_adaptation_speed": avg_adaptation,
            "average_context_sensitivity": avg_context,
            "num_instructions_tested": len(instructions)
        }
        
        return EvaluationResult(
            task_name="Efficient_IGPM_Plasticity",
            score=overall_score,
            metrics=metrics,
            execution_time=execution_time,
            memory_usage=memory_usage,
            model_info={"model_type": "efficient_core_nn", "parameters": sum(p.numel() for p in self.model.parameters())},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def evaluate_glue(self) -> EvaluationResult:
        """Evaluate GLUE tasks using the same methodology as the original framework."""
        print("Evaluating Efficient CORE-NN on GLUE-style Tasks...")
        
        start_time = time.time()
        start_memory = self._measure_memory_usage()
        
        # GLUE task examples (same as original)
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
            "wnli": [
                {
                    "sentence1": "The trophy doesn't fit in the suitcase because it's too big.",
                    "sentence2": "The trophy is too big.",
                    "label": "entailment",
                    "instruction": "Does sentence 2 follow from sentence 1?"
                },
                {
                    "sentence1": "The trophy doesn't fit in the suitcase because it's too small.",
                    "sentence2": "The trophy is too small.",
                    "label": "not_entailment",
                    "instruction": "Does sentence 2 follow from sentence 1?"
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
        
        task_scores = []
        task_results = {}
        
        for task_name, examples in glue_tasks.items():
            # Collect training data
            training_data = []
            
            for example in examples:
                # Reset model position
                self.model.current_position = 0
                
                # Create input text
                if task_name == "rte":
                    input_text = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
                elif task_name == "wnli":
                    input_text = f"Sentence 1: {example['sentence1']} Sentence 2: {example['sentence2']}"
                else:  # sentiment
                    input_text = example['text']
                
                input_ids = self._tokenize_text(input_text)
                
                with torch.no_grad():
                    output = self.model.forward(input_ids, instruction=example['instruction'])
                    hidden_states = output["last_hidden_state"]
                    training_data.append((hidden_states, example['label']))
            
            # Train classification head
            _ = self.classification_manager.train_head(
                task_name=task_name,
                examples=training_data,
                learning_rate=1e-3,
                epochs=20
            )
            
            # Evaluate
            evaluation_info = self.classification_manager.evaluate_head(task_name, training_data)
            accuracy = evaluation_info["accuracy"]
            
            task_scores.append(accuracy)
            task_results[f"{task_name}_score"] = accuracy
        
        # Calculate overall score
        overall_score = float(np.mean(task_scores))
        task_results["average_score"] = overall_score
        task_results["num_tasks"] = len(task_scores)
        
        execution_time = time.time() - start_time
        memory_usage = self._measure_memory_usage() - start_memory
        
        return EvaluationResult(
            task_name="Efficient_GLUE_Benchmark",
            score=overall_score,
            metrics=task_results,
            execution_time=execution_time,
            memory_usage=memory_usage,
            model_info={"model_type": "efficient_core_nn", "parameters": sum(p.numel() for p in self.model.parameters())},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def run_full_evaluation(self) -> Dict[str, EvaluationResult]:
        """Run full evaluation suite."""
        print("Starting Efficient CORE-NN Evaluation")
        print("=" * 50)
        
        results = {}
        
        # Run plasticity evaluation
        plasticity_result = self.evaluate_plasticity()
        results["plasticity"] = plasticity_result
        print(f"✅ Plasticity evaluation completed: score={plasticity_result.score:.4f}")
        
        # Run GLUE evaluation
        glue_result = self.evaluate_glue()
        results["glue"] = glue_result
        print(f"✅ GLUE evaluation completed: score={glue_result.score:.4f}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("optimization/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"efficient_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({k: v.__dict__ for k, v in results.items()}, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / f"efficient_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("Efficient CORE-NN Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for name, result in results.items():
                f.write(f"{name.upper()} EVALUATION:\n")
                f.write(f"  Score: {result.score:.4f}\n")
                f.write(f"  Execution Time: {result.execution_time:.2f}s\n")
                f.write(f"  Memory Usage: {result.memory_usage:.2f}MB\n")
                f.write(f"  Model Parameters: {result.model_info.get('parameters', 'N/A'):,}\n")
                f.write(f"  Key Metrics:\n")
                for metric, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"    {metric}: {value:.4f}\n")
                f.write("\n")
        
        print(f"\nResults saved to {results_file}")
        print(f"Summary saved to {summary_file}")
        
        return results


def main():
    """Run efficient model evaluation."""
    evaluator = EfficientModelEvaluator()
    results = evaluator.run_full_evaluation()
    
    print(f"\n🎯 EFFICIENT CORE-NN EVALUATION SUMMARY:")
    for name, result in results.items():
        print(f"  {name}: {result.score:.4f}")
    
    # Compare with original results
    print(f"\n📊 COMPARISON WITH ORIGINAL CORE-NN:")
    print(f"Original Plasticity: 0.5125 | Efficient: {results['plasticity'].score:.4f}")
    print(f"Original GLUE: 0.6111 | Efficient: {results['glue'].score:.4f}")
    
    return results


if __name__ == "__main__":
    main()
