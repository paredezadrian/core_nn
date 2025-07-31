"""
Baseline Comparison Framework for CORE-NN vs Transformer.

This module implements head-to-head comparison between CORE-NN and 
a standard transformer baseline on GLUE benchmark tasks.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import sys

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from evaluation.transformer_baseline import create_transformer_baseline, TransformerBaseline
from evaluation.classification_heads import GLUEClassificationManager
from evaluation.evaluation_framework import EvaluationConfig, EvaluationResult


@dataclass
class ComparisonResult:
    """Results from baseline comparison."""
    core_nn_result: EvaluationResult
    transformer_result: EvaluationResult
    comparison_metrics: Dict[str, Any]
    timestamp: str


class BaselineComparator:
    """Compares CORE-NN against transformer baseline on GLUE tasks."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CORE-NN model
        self.core_nn_config = ConfigManager().load_config(config.model_config_path)
        self.core_nn_model = CoreNNModel(self.core_nn_config, vocab_size=50000)
        self.core_nn_model.to(self.device)
        self.core_nn_model.eval()
        
        # Create transformer baseline with similar complexity
        self.transformer_model = create_transformer_baseline(
            embedding_dim=self.core_nn_config.rteu.embedding_dim,
            num_layers=6,  # Comparable to CORE-NN component complexity
            vocab_size=50000
        )
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        
        # Initialize classification managers for both models
        self.core_nn_classifier = GLUEClassificationManager(
            input_dim=self.core_nn_config.rteu.embedding_dim,
            device=str(self.device)
        )
        
        self.transformer_classifier = GLUEClassificationManager(
            input_dim=self.core_nn_config.rteu.embedding_dim,
            device=str(self.device)
        )
        
        print(f"Loaded models on {self.device}")
        print(f"CORE-NN parameters: {sum(p.numel() for p in self.core_nn_model.parameters()):,}")
        print(f"Transformer parameters: {self.transformer_model.get_parameter_count():,}")
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple tokenization for evaluation (character-based for now)."""
        # Limit text length to avoid position embedding issues
        max_len = min(self.config.max_sequence_length, 50)
        
        # Convert to character IDs (ASCII values, capped at vocab size)
        char_ids = [min(ord(c), 49999) for c in text[:max_len]]
        
        # Ensure minimum length
        if len(char_ids) < 5:
            char_ids.extend([0] * (5 - len(char_ids)))
        
        return torch.tensor([char_ids], dtype=torch.long, device=self.device)
    
    def _evaluate_model_on_glue(self, 
                               model: torch.nn.Module, 
                               classifier: GLUEClassificationManager,
                               model_name: str) -> Dict[str, float]:
        """Evaluate a model on GLUE tasks."""
        print(f"\nEvaluating {model_name} on GLUE tasks...")
        
        # GLUE task examples (same as in evaluation framework)
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
        
        task_results = {}
        
        for task_name, examples in glue_tasks.items():
            print(f"  Running {task_name}...")
            
            # Collect training data
            training_data = []
            
            for example in examples:
                # Reset model position for CORE-NN
                if hasattr(model, 'current_position'):
                    model.current_position = 0
                
                # Create input text based on task
                if task_name == "rte":
                    input_text = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
                elif task_name == "wnli":
                    input_text = f"Sentence 1: {example['sentence1']} Sentence 2: {example['sentence2']}"
                else:  # sentiment
                    input_text = example['text']
                
                input_ids = self._tokenize_text(input_text)
                
                with torch.no_grad():
                    # Get model output
                    if isinstance(model, CoreNNModel):
                        output = model.forward(input_ids, instruction=example['instruction'])
                    else:  # Transformer
                        output = model.forward(input_ids, instruction=example['instruction'])
                    
                    hidden_states = output["last_hidden_state"]
                    training_data.append((hidden_states, example['label']))
            
            # Train classification head
            _ = classifier.train_head(
                task_name=task_name,
                examples=training_data,
                learning_rate=1e-3,
                epochs=20
            )
            
            # Evaluate
            evaluation_info = classifier.evaluate_head(task_name, training_data)
            task_results[task_name] = evaluation_info["accuracy"]
            
            print(f"    {task_name}: {evaluation_info['accuracy']:.4f}")
        
        # Calculate overall score
        overall_score = np.mean(list(task_results.values()))
        task_results["overall"] = overall_score
        
        return task_results
    
    def run_comparison(self) -> ComparisonResult:
        """Run head-to-head comparison between CORE-NN and transformer baseline."""
        print("Starting CORE-NN vs Transformer Baseline Comparison")
        print("=" * 60)
        
        start_time = time.time()
        
        # Evaluate CORE-NN
        core_nn_start = time.time()
        core_nn_results = self._evaluate_model_on_glue(
            self.core_nn_model, 
            self.core_nn_classifier, 
            "CORE-NN"
        )
        core_nn_time = time.time() - core_nn_start
        
        # Evaluate Transformer
        transformer_start = time.time()
        transformer_results = self._evaluate_model_on_glue(
            self.transformer_model, 
            self.transformer_classifier, 
            "Transformer"
        )
        transformer_time = time.time() - transformer_start
        
        total_time = time.time() - start_time
        
        # Create comparison metrics
        comparison_metrics = {
            "core_nn_overall": core_nn_results["overall"],
            "transformer_overall": transformer_results["overall"],
            "performance_difference": core_nn_results["overall"] - transformer_results["overall"],
            "relative_improvement": ((core_nn_results["overall"] - transformer_results["overall"]) / transformer_results["overall"] * 100) if transformer_results["overall"] > 0 else 0,
            "core_nn_time": core_nn_time,
            "transformer_time": transformer_time,
            "speed_ratio": transformer_time / core_nn_time if core_nn_time > 0 else 1,
            "task_breakdown": {
                task: {
                    "core_nn": core_nn_results.get(task, 0),
                    "transformer": transformer_results.get(task, 0),
                    "difference": core_nn_results.get(task, 0) - transformer_results.get(task, 0)
                }
                for task in ["rte", "wnli", "sentiment"]
            }
        }
        
        # Create evaluation results
        core_nn_result = EvaluationResult(
            task_name="CORE-NN_GLUE_Comparison",
            score=core_nn_results["overall"],
            metrics=core_nn_results,
            execution_time=core_nn_time,
            memory_usage=0.0,  # Simplified for comparison
            model_info={"model_type": "core_nn", "enhanced_igpm": True},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        transformer_result = EvaluationResult(
            task_name="Transformer_GLUE_Comparison",
            score=transformer_results["overall"],
            metrics=transformer_results,
            execution_time=transformer_time,
            memory_usage=0.0,  # Simplified for comparison
            model_info=self.transformer_model.get_model_info(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return ComparisonResult(
            core_nn_result=core_nn_result,
            transformer_result=transformer_result,
            comparison_metrics=comparison_metrics,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def save_comparison_results(self, result: ComparisonResult, output_dir: Path):
        """Save comparison results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_dir / f"baseline_comparison_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / f"comparison_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("CORE-NN vs Transformer Baseline Comparison\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"CORE-NN Overall Score: {result.comparison_metrics['core_nn_overall']:.4f}\n")
            f.write(f"Transformer Overall Score: {result.comparison_metrics['transformer_overall']:.4f}\n")
            f.write(f"Performance Difference: {result.comparison_metrics['performance_difference']:.4f}\n")
            f.write(f"Relative Improvement: {result.comparison_metrics['relative_improvement']:.2f}%\n\n")
            
            f.write("Task Breakdown:\n")
            for task, metrics in result.comparison_metrics['task_breakdown'].items():
                f.write(f"  {task.upper()}:\n")
                f.write(f"    CORE-NN: {metrics['core_nn']:.4f}\n")
                f.write(f"    Transformer: {metrics['transformer']:.4f}\n")
                f.write(f"    Difference: {metrics['difference']:.4f}\n\n")
            
            f.write(f"Execution Times:\n")
            f.write(f"  CORE-NN: {result.comparison_metrics['core_nn_time']:.2f}s\n")
            f.write(f"  Transformer: {result.comparison_metrics['transformer_time']:.2f}s\n")
            f.write(f"  Speed Ratio: {result.comparison_metrics['speed_ratio']:.2f}x\n")
        
        print(f"\nComparison results saved to {results_file}")
        print(f"Summary saved to {summary_file}")
        
        return results_file, summary_file


if __name__ == "__main__":
    # Run baseline comparison
    config = EvaluationConfig()
    comparator = BaselineComparator(config)
    
    result = comparator.run_comparison()
    
    # Save results
    output_dir = Path("evaluation/results")
    comparator.save_comparison_results(result, output_dir)
    
    # Print summary
    print(f"\n🎯 COMPARISON SUMMARY:")
    print(f"CORE-NN: {result.comparison_metrics['core_nn_overall']:.4f}")
    print(f"Transformer: {result.comparison_metrics['transformer_overall']:.4f}")
    print(f"Improvement: {result.comparison_metrics['relative_improvement']:.2f}%")
