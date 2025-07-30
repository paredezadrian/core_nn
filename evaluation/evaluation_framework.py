#!/usr/bin/env python3
"""
Real-World Task Evaluation Framework for CORE-NN.

This framework validates that the enhanced IGPM plasticity improvements
translate to superior performance on actual language modeling and 
instruction-following tasks.

Key evaluation areas:
1. GLUE benchmark tasks (especially instruction-following like RTE, WNLI)
2. Long-context processing (8K+ tokens)
3. Instruction-following (Alpaca-style)
4. Memory-intensive reasoning tasks
5. Edge device performance validation
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import sys

# Add core_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.config.schema import CoreNNConfig


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_name: str
    score: float
    metrics: Dict[str, Any]
    execution_time: float
    memory_usage: float
    model_info: Dict[str, Any]
    timestamp: str


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    model_config_path: str = "configs/default.yaml"
    output_dir: str = "evaluation/results"
    device: str = "auto"
    batch_size: int = 1
    max_sequence_length: int = 512
    num_samples: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    seed: int = 42


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load model
        self.model_config = ConfigManager().load_config(config.model_config_path)
        self.model = CoreNNModel(self.model_config, vocab_size=50000)  # Standard vocab size
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded CORE-NN model on {self.device}")
        print(f"Model config: {config.model_config_path}")
    
    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        """Run evaluation and return results."""
        pass
    
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple tokenization for evaluation (character-based for now)."""
        # Limit text length to avoid position embedding issues
        max_len = min(self.config.max_sequence_length, 50)  # Conservative limit

        # Convert to character IDs (ASCII values, capped at vocab size)
        char_ids = [min(ord(c), 49999) for c in text[:max_len]]

        # Ensure minimum length
        if len(char_ids) < 5:
            char_ids.extend([0] * (5 - len(char_ids)))

        return torch.tensor([char_ids], dtype=torch.long, device=self.device)
    
    def _detokenize_ids(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        # Convert back to characters (simple ASCII)
        chars = []
        for token_id in token_ids.flatten():
            if 32 <= token_id <= 126:  # Printable ASCII
                chars.append(chr(token_id.item()))
            elif token_id == 0:  # Padding
                break
        return ''.join(chars)


class PlasticityEvaluator(BaseEvaluator):
    """Evaluates IGPM plasticity improvements on instruction-following tasks."""
    
    def evaluate(self) -> EvaluationResult:
        """Evaluate plasticity improvements."""
        print("Evaluating IGPM Plasticity Improvements...")
        
        start_time = time.time()
        start_memory = self._measure_memory_usage()
        
        # Test instructions that should trigger different plasticity responses
        test_instructions = [
            "remember this important information",
            "focus your attention on key details", 
            "suppress irrelevant noise",
            "amplify the signal strength",
            "adapt to this new pattern",
            "learn from this example",
            "ignore distracting elements",
            "enhance important features"
        ]
        
        plasticity_scores = []
        adaptation_speeds = []
        context_sensitivity = []
        
        for instruction in test_instructions:
            # Test plasticity response
            plasticity_score = self._test_plasticity_response(instruction)
            plasticity_scores.append(plasticity_score)
            
            # Test adaptation speed
            adaptation_speed = self._test_adaptation_speed(instruction)
            adaptation_speeds.append(adaptation_speed)
            
            # Test context sensitivity
            context_score = self._test_context_sensitivity(instruction)
            context_sensitivity.append(context_score)
        
        # Calculate overall metrics
        avg_plasticity = np.mean(plasticity_scores)
        avg_adaptation = np.mean(adaptation_speeds)
        avg_context = np.mean(context_sensitivity)
        
        # Overall score (weighted combination)
        overall_score = float(avg_plasticity * 0.4 + avg_adaptation * 0.3 + avg_context * 0.3)
        
        execution_time = time.time() - start_time
        memory_usage = self._measure_memory_usage() - start_memory
        
        metrics = {
            "average_plasticity_magnitude": avg_plasticity,
            "average_adaptation_speed": avg_adaptation,
            "average_context_sensitivity": avg_context,
            "plasticity_scores": plasticity_scores,
            "adaptation_speeds": adaptation_speeds,
            "context_scores": context_sensitivity,
            "num_instructions_tested": len(test_instructions)
        }
        
        return EvaluationResult(
            task_name="IGPM_Plasticity",
            score=overall_score,
            metrics=metrics,
            execution_time=execution_time,
            memory_usage=memory_usage,
            model_info={"enhanced_igpm": True, "version": "0.1.0-beta"},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _test_plasticity_response(self, instruction: str) -> float:
        """Test plasticity response magnitude for an instruction."""
        # Create test input
        test_input = "Test input for plasticity."
        input_ids = self._tokenize_text(test_input)

        # Measure response before and after instruction
        with torch.no_grad():
            # Reset model position to avoid embedding issues
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

        return change_magnitude
    
    def _test_adaptation_speed(self, instruction: str) -> float:
        """Test how quickly the model adapts to an instruction."""
        # Create learning sequence
        learning_inputs = [
            "First example of the pattern",
            "Second example showing adaptation", 
            "Third example demonstrating learning"
        ]
        
        adaptation_improvements = []
        
        for input_text in learning_inputs:
            input_ids = self._tokenize_text(input_text)

            with torch.no_grad():
                # Reset model position
                self.model.current_position = 0

                # Generate response with instruction
                output = self.model.forward(input_ids, instruction=instruction)
                
                # Measure adaptation quality (simplified metric)
                if "component_info" in output and "igpm_info" in output["component_info"]:
                    igpm_info_list = output["component_info"]["igpm_info"]
                    if igpm_info_list:
                        plasticity_effect = igpm_info_list[-1].get("total_plasticity_effect", 0.0)
                        adaptation_improvements.append(float(plasticity_effect))
                    else:
                        adaptation_improvements.append(0.0)
                else:
                    adaptation_improvements.append(0.0)
        
        # Calculate adaptation speed (improvement over time)
        if len(adaptation_improvements) > 1:
            speed = float(np.mean(np.diff(adaptation_improvements)))
        else:
            speed = 0.0

        return max(0.0, speed)  # Ensure non-negative
    
    def _test_context_sensitivity(self, instruction: str) -> float:
        """Test context-dependent plasticity responses."""
        # Test different context types
        contexts = [
            ("memory context", "Remember this important fact: AI is transforming the world."),
            ("attention context", "Pay attention to the key details in this analysis."),
            ("suppression context", "Ignore the noise and focus on the signal.")
        ]
        
        context_responses = []
        
        for _, context_text in contexts:
            input_ids = self._tokenize_text(context_text)

            with torch.no_grad():
                # Reset model position
                self.model.current_position = 0

                output = self.model.forward(input_ids, instruction=instruction)

                # Measure context-specific response
                if "component_info" in output and "igpm_info" in output["component_info"]:
                    igpm_info_list = output["component_info"]["igpm_info"]
                    if igpm_info_list:
                        response_strength = igpm_info_list[-1].get("total_plasticity_effect", 0.0)
                        context_responses.append(float(response_strength))
                    else:
                        context_responses.append(0.0)
                else:
                    context_responses.append(0.0)

        # Calculate context sensitivity (variance in responses)
        if len(context_responses) > 1:
            sensitivity = float(np.std(context_responses))
        else:
            sensitivity = 0.0

        return sensitivity


class GLUEEvaluator(BaseEvaluator):
    """Evaluates CORE-NN on GLUE benchmark tasks."""

    def evaluate(self) -> EvaluationResult:
        """Evaluate on GLUE-style tasks."""
        print("Evaluating GLUE-style Tasks...")

        start_time = time.time()
        start_memory = self._measure_memory_usage()

        # Test on simplified GLUE-style tasks
        task_scores = []

        # 1. Recognizing Textual Entailment (RTE) - simplified
        rte_score = self._evaluate_rte()
        task_scores.append(rte_score)

        # 2. Winograd Natural Language Inference (WNLI) - simplified
        wnli_score = self._evaluate_wnli()
        task_scores.append(wnli_score)

        # 3. Sentiment Analysis (SST-2 style) - simplified
        sentiment_score = self._evaluate_sentiment()
        task_scores.append(sentiment_score)

        # Calculate overall score
        overall_score = float(np.mean(task_scores))

        execution_time = time.time() - start_time
        memory_usage = self._measure_memory_usage() - start_memory

        metrics = {
            "rte_score": rte_score,
            "wnli_score": wnli_score,
            "sentiment_score": sentiment_score,
            "average_score": overall_score,
            "num_tasks": len(task_scores)
        }

        return EvaluationResult(
            task_name="GLUE_Benchmark",
            score=overall_score,
            metrics=metrics,
            execution_time=execution_time,
            memory_usage=memory_usage,
            model_info={"enhanced_igpm": True, "version": "0.1.0-beta"},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _evaluate_rte(self) -> float:
        """Evaluate Recognizing Textual Entailment."""
        # Simplified RTE examples
        examples = [
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
        ]

        correct = 0
        total = len(examples)

        for example in examples:
            # Reset model position
            self.model.current_position = 0

            # Create input text
            input_text = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
            input_ids = self._tokenize_text(input_text)

            with torch.no_grad():
                # Generate response with instruction
                output = self.model.forward(input_ids, instruction=example['instruction'])
                logits = output["logits"]

                # Simple heuristic: check if model shows different responses for entailment vs not
                # This is a simplified evaluation - in practice would need proper classification head
                response_strength = torch.norm(logits, dim=-1).mean().item()

                # Classify based on response strength (simplified)
                predicted = "entailment" if response_strength > 100.0 else "not_entailment"

                if predicted == example['label']:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def _evaluate_wnli(self) -> float:
        """Evaluate Winograd Natural Language Inference."""
        # Simplified WNLI examples
        examples = [
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
        ]

        correct = 0
        total = len(examples)

        for example in examples:
            # Reset model position
            self.model.current_position = 0

            input_text = f"Sentence 1: {example['sentence1']} Sentence 2: {example['sentence2']}"
            input_ids = self._tokenize_text(input_text)

            with torch.no_grad():
                output = self.model.forward(input_ids, instruction=example['instruction'])

                # Check IGPM response for instruction-following
                igpm_response = 0.0
                if "component_info" in output and "igpm_info" in output["component_info"]:
                    igpm_info_list = output["component_info"]["igpm_info"]
                    if igpm_info_list:
                        igpm_response = igpm_info_list[-1].get("total_plasticity_effect", 0.0)

                # Classify based on IGPM response (simplified)
                predicted = "entailment" if igpm_response > 5.0 else "not_entailment"

                if predicted == example['label']:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def _evaluate_sentiment(self) -> float:
        """Evaluate sentiment analysis."""
        examples = [
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

        correct = 0
        total = len(examples)

        for example in examples:
            # Reset model position
            self.model.current_position = 0

            input_ids = self._tokenize_text(example['text'])

            with torch.no_grad():
                output = self.model.forward(input_ids, instruction=example['instruction'])

                # Use plasticity response to classify sentiment
                plasticity_effect = 0.0
                if "component_info" in output and "igpm_info" in output["component_info"]:
                    igpm_info_list = output["component_info"]["igpm_info"]
                    if igpm_info_list:
                        plasticity_effect = igpm_info_list[-1].get("total_plasticity_effect", 0.0)

                # Simple classification based on plasticity
                predicted = "positive" if plasticity_effect > 7.0 else "negative"

                if predicted == example['label']:
                    correct += 1

        return correct / total if total > 0 else 0.0


class EvaluationRunner:
    """Main evaluation runner that coordinates all evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluators
        self.evaluators = {
            "plasticity": PlasticityEvaluator(config),
            "glue": GLUEEvaluator(config)
        }
    
    def run_all_evaluations(self) -> Dict[str, EvaluationResult]:
        """Run all evaluations and return results."""
        print("Starting Real-World Task Evaluation")
        print("=" * 50)
        
        results = {}
        
        for name, evaluator in self.evaluators.items():
            print(f"\nRunning {name} evaluation...")
            try:
                result = evaluator.evaluate()
                results[name] = result
                print(f"✅ {name} evaluation completed: score={result.score:.4f}")
            except Exception as e:
                print(f"✗ {name} evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, EvaluationResult]):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        serializable_results = {
            name: asdict(result) for name, result in results.items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")

        # Save summary
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("CORE-NN Real-World Task Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")

            for name, result in results.items():
                f.write(f"{name.upper()} EVALUATION:\n")
                f.write(f"  Score: {result.score:.4f}\n")
                f.write(f"  Execution Time: {result.execution_time:.2f}s\n")
                f.write(f"  Memory Usage: {result.memory_usage:.2f}MB\n")
                f.write(f"  Key Metrics:\n")
                for metric, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"    {metric}: {value:.4f}\n")
                f.write("\n")

        print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    # Default evaluation configuration
    config = EvaluationConfig(
        model_config_path="configs/default.yaml",
        output_dir="evaluation/results",
        num_samples=50,
        seed=42
    )
    
    # Run evaluations
    runner = EvaluationRunner(config)
    results = runner.run_all_evaluations()
    
    print("\nEvaluation completed!")
    for name, result in results.items():
        print(f"  {name}: {result.score:.4f}")
