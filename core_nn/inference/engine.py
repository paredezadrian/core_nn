"""
Inference Engine for CORE-NN.

Handles model inference, generation, and runtime optimization.
"""

import torch
import torch.nn.functional as F
import time
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import numpy as np

from ..config.schema import InferenceConfig, CoreNNConfig
from ..model import CoreNNModel
from ..memory.episodic_store import EpisodicStore


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 0
    use_cache: bool = True


@dataclass
class InferenceResult:
    """Result from inference operation."""
    generated_ids: torch.Tensor
    generated_text: str
    logits: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    component_info: Optional[Dict[str, Any]] = None


class InferenceEngine:
    """
    Inference Engine for CORE-NN.
    
    Provides optimized inference capabilities including:
    - Text generation with various sampling strategies
    - Batch processing
    - Memory optimization
    - Performance monitoring
    """
    
    def __init__(self,
                 model: CoreNNModel,
                 config: InferenceConfig,
                 tokenizer: Optional[Any] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        # Initialize episodic store for memory operations
        self.episodic_store = EpisodicStore(
            bcm=model.bcm,
            igpm=model.igpm,
            enable_disk_cache=False  # Can be made configurable
        )

        # Performance tracking
        self.inference_stats = {
            "total_inferences": 0,
            "total_tokens_generated": 0,
            "average_tokens_per_second": 0.0,
            "average_memory_usage": 0.0,
            "error_count": 0
        }

        # Optimization settings
        self.use_mixed_precision = hasattr(model, 'config') and model.config.device.mixed_precision
        self.memory_efficient = hasattr(model, 'config') and model.config.device.memory_efficient

        # Setup model for inference
        self.model.eval()
        
    def generate(self,
                input_text: Optional[str] = None,
                input_ids: Optional[torch.Tensor] = None,
                generation_config: Optional[GenerationConfig] = None,
                instruction: Optional[str] = None,
                return_full_output: bool = False) -> InferenceResult:
        """
        Generate text using CORE-NN model.

        Args:
            input_text: Input text string
            input_ids: Input token IDs tensor
            generation_config: Generation configuration
            instruction: Optional instruction for guided generation
            return_full_output: Whether to return full model outputs

        Returns:
            InferenceResult with generated text and metadata
        """
        start_time = time.time()
        initial_memory = self._get_memory_usage()

        try:
            # Setup generation config
            if generation_config is None:
                generation_config = GenerationConfig(
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty
                )

            # Prepare input
            if input_ids is None:
                if input_text is None:
                    raise ValueError("Either input_text or input_ids must be provided")
                input_ids = self._tokenize(input_text)

            # Check for system commands before normal generation
            system_command_result = self._handle_system_commands(input_text, input_ids)
            if system_command_result is not None:
                return system_command_result

            # Ensure input is on correct device
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Generate with model
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.autocast(device_type=device.type):
                        result = self._generate_tokens(input_ids, generation_config, instruction)
                else:
                    result = self._generate_tokens(input_ids, generation_config, instruction)
            
            # Calculate performance metrics
            generation_time = time.time() - start_time
            num_generated_tokens = result['generated_ids'].size(1) - input_ids.size(1)
            tokens_per_second = num_generated_tokens / generation_time if generation_time > 0 else 0
            memory_usage = self._get_memory_usage() - initial_memory
            
            # Detokenize generated text
            generated_text = self._detokenize(result['generated_ids'])
            
            # Update statistics
            self._update_stats(num_generated_tokens, tokens_per_second, memory_usage)
            
            # Prepare result
            inference_result = InferenceResult(
                generated_ids=result['generated_ids'],
                generated_text=generated_text,
                logits=result.get('logits') if return_full_output else None,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
                memory_usage_mb=memory_usage,
                component_info=result.get('component_info') if return_full_output else None
            )
            
            return inference_result
            
        except Exception as e:
            self.inference_stats["error_count"] += 1
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _generate_tokens(self, 
                        input_ids: torch.Tensor,
                        generation_config: GenerationConfig,
                        instruction: Optional[str] = None) -> Dict[str, Any]:
        """Generate tokens using the model."""
        batch_size, seq_len = input_ids.shape
        max_length = seq_len + generation_config.max_new_tokens
        
        # Initialize generation
        generated_ids = input_ids.clone()
        past_key_values = None
        all_logits = []
        component_info = []
        
        # Generation loop
        for step in range(generation_config.max_new_tokens):
            # Prepare input for current step
            if generation_config.use_cache and past_key_values is not None:
                # Only use last token if using cache
                current_input = generated_ids[:, -1:]
            else:
                current_input = generated_ids
            
            # Forward pass
            outputs = self.model.forward(
                current_input,
                instruction=instruction,
                reset_state=(step == 0)
            )
            
            # Get logits for next token prediction
            logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
            all_logits.append(logits)
            component_info.append(outputs.get("component_info"))
            
            # Apply generation parameters
            next_token_logits = self._apply_generation_config(
                logits, generated_ids, generation_config
            )
            
            # Sample next token
            if generation_config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for early stopping
            if (generation_config.early_stopping and 
                next_token.item() == generation_config.eos_token_id):
                break
            
            # Check max length
            if generated_ids.size(1) >= max_length:
                break
        
        return {
            "generated_ids": generated_ids,
            "logits": torch.stack(all_logits, dim=1) if all_logits else None,
            "component_info": component_info
        }
    
    def _apply_generation_config(self, 
                               logits: torch.Tensor,
                               generated_ids: torch.Tensor,
                               config: GenerationConfig) -> torch.Tensor:
        """Apply generation configuration to logits."""
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()):
                logits[0, token_id] /= config.repetition_penalty
        
        # Apply top-k filtering
        if config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_k_indices, top_k_logits)
        
        # Apply top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return logits

    def _handle_system_commands(self,
                               input_text: Optional[str],
                               input_ids: torch.Tensor) -> Optional[InferenceResult]:
        """
        Handle system commands like #remember, #recall, #forget.

        Args:
            input_text: Original input text
            input_ids: Tokenized input

        Returns:
            InferenceResult if system command was handled, None otherwise
        """
        if input_text is None:
            return None

        # Check for system commands in text
        text = input_text.strip()

        # Parse #remember("term") or #remember(term)
        if text.startswith('#remember'):
            return self._handle_remember_command(text)

        # Parse #recall("term") or #recall(term)
        elif text.startswith('#recall'):
            return self._handle_recall_command(text)

        # Parse #forget("term") or #forget(term)
        elif text.startswith('#forget'):
            return self._handle_forget_command(text)

        # Check if tokenized input contains system command tokens
        if self.tokenizer and hasattr(self.tokenizer, 'vocabulary'):
            for token_id in input_ids[0]:
                token = self.tokenizer.vocabulary.get_token(token_id.item())
                if token and token.startswith('#'):
                    if token == '#remember':
                        return self._handle_remember_command_from_tokens(input_ids)
                    elif token == '#recall':
                        return self._handle_recall_command_from_tokens(input_ids)
                    elif token == '#forget':
                        return self._handle_forget_command_from_tokens(input_ids)

        return None

    def _handle_remember_command(self, text: str) -> InferenceResult:
        """Handle #remember command from text."""
        # Extract content from #remember("content") or #remember(content)
        import re
        match = re.search(r'#remember\s*\(\s*["\']?([^"\'()]+)["\']?\s*\)', text)
        if match:
            content = match.group(1).strip()
        else:
            # Fallback: everything after #remember
            content = text[9:].strip()  # Remove '#remember'

        if not content:
            content = "Empty remember command"

        # Use episodic store to remember
        result = self.episodic_store.remember(content, content)

        # Create response text
        response_text = f"Remembered: {content}"

        # Create fake token IDs for response
        response_ids = self._tokenize(response_text)

        return InferenceResult(
            generated_ids=response_ids,
            generated_text=response_text,
            generation_time=0.01,
            tokens_per_second=len(response_ids) / 0.01,
            memory_usage_mb=0.0,
            component_info={"system_command": "remember", "result": result}
        )

    def _handle_recall_command(self, text: str) -> InferenceResult:
        """Handle #recall command from text."""
        # Extract query from #recall("query") or #recall(query)
        import re
        match = re.search(r'#recall\s*\(\s*["\']?([^"\'()]+)["\']?\s*\)', text)
        if match:
            query = match.group(1).strip()
        else:
            # Fallback: everything after #recall
            query = text[7:].strip()  # Remove '#recall'

        if not query:
            query = "empty query"

        # Use episodic store to recall
        result = self.episodic_store.recall(query)

        # Create response text
        total_found = result.get('total_found', 0)
        if total_found > 0:
            memories = []
            for mem in result.get('local_memories', [])[:3]:
                memories.append(f"- {mem['content']}")
            for mem in result.get('igpm_memories', [])[:3]:
                memories.append(f"- {mem['instruction']}")

            response_text = f"Recalled {total_found} memories for '{query}':\n" + "\n".join(memories)
        else:
            response_text = f"No memories found for '{query}'"

        # Create fake token IDs for response
        response_ids = self._tokenize(response_text)

        return InferenceResult(
            generated_ids=response_ids,
            generated_text=response_text,
            generation_time=0.01,
            tokens_per_second=len(response_ids) / 0.01,
            memory_usage_mb=0.0,
            component_info={"system_command": "recall", "result": result}
        )

    def _handle_forget_command(self, text: str) -> InferenceResult:
        """Handle #forget command from text."""
        # Extract query from #forget("query") or #forget(query)
        import re
        match = re.search(r'#forget\s*\(\s*["\']?([^"\'()]+)["\']?\s*\)', text)
        if match:
            query = match.group(1).strip()
        else:
            # Fallback: everything after #forget
            query = text[7:].strip()  # Remove '#forget'

        if not query:
            query = "empty query"

        # Use episodic store to forget
        result = self.episodic_store.forget(query)

        # Create response text
        total_removed = sum(result[k] for k in result if k.endswith('_removed'))
        response_text = f"Forgot {total_removed} memories related to '{query}'"

        # Create fake token IDs for response
        response_ids = self._tokenize(response_text)

        return InferenceResult(
            generated_ids=response_ids,
            generated_text=response_text,
            generation_time=0.01,
            tokens_per_second=len(response_ids) / 0.01,
            memory_usage_mb=0.0,
            component_info={"system_command": "forget", "result": result}
        )

    def _handle_remember_command_from_tokens(self, input_ids: torch.Tensor) -> InferenceResult:
        """Handle #remember command from tokenized input."""
        # Simple implementation - detokenize and parse
        text = self._detokenize(input_ids)
        return self._handle_remember_command(text)

    def _handle_recall_command_from_tokens(self, input_ids: torch.Tensor) -> InferenceResult:
        """Handle #recall command from tokenized input."""
        # Simple implementation - detokenize and parse
        text = self._detokenize(input_ids)
        return self._handle_recall_command(text)

    def _handle_forget_command_from_tokens(self, input_ids: torch.Tensor) -> InferenceResult:
        """Handle #forget command from tokenized input."""
        # Simple implementation - detokenize and parse
        text = self._detokenize(input_ids)
        return self._handle_forget_command(text)

    def batch_generate(self,
                      inputs: List[Union[str, torch.Tensor]],
                      generation_config: Optional[GenerationConfig] = None,
                      instructions: Optional[List[str]] = None) -> List[InferenceResult]:
        """Generate text for multiple inputs in batch."""
        results = []
        
        for i, input_item in enumerate(inputs):
            instruction = instructions[i] if instructions and i < len(instructions) else None
            
            if isinstance(input_item, str):
                result = self.generate(
                    input_text=input_item,
                    generation_config=generation_config,
                    instruction=instruction
                )
            else:
                result = self.generate(
                    input_ids=input_item,
                    generation_config=generation_config,
                    instruction=instruction
                )
            
            results.append(result)
        
        return results
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text."""
        if self.tokenizer:
            # Use provided tokenizer (ASCTokenizer uses tokenize method)
            if hasattr(self.tokenizer, 'tokenize'):
                tokens = self.tokenizer.tokenize(text, add_special_tokens=True)
                tokens = torch.tensor([tokens], dtype=torch.long)
            else:
                # Fallback for other tokenizer types
                tokens = self.tokenizer.encode(text, return_tensors="pt")
        else:
            # Simple character-based tokenization (fallback)
            tokens = [min(ord(c), 999) for c in text[:self.config.max_sequence_length]]
            tokens = torch.tensor([tokens], dtype=torch.long)

        return tokens
    
    def _detokenize(self, token_ids: torch.Tensor) -> str:
        """Detokenize token IDs to text."""
        if self.tokenizer:
            # Use provided tokenizer (ASCTokenizer uses detokenize method)
            if hasattr(self.tokenizer, 'detokenize'):
                token_list = token_ids[0].tolist()
                return self.tokenizer.detokenize(token_list, skip_special_tokens=True)
            else:
                # Fallback for other tokenizer types
                return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        else:
            # Simple character-based detokenization (fallback)
            chars = [chr(min(max(int(token.item()), 32), 126)) for token in token_ids[0] if token.item() > 0]
            return ''.join(chars)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def _update_stats(self, num_tokens: int, tokens_per_second: float, memory_usage: float):
        """Update inference statistics."""
        self.inference_stats["total_inferences"] += 1
        self.inference_stats["total_tokens_generated"] += num_tokens
        
        # Update running averages
        total_inferences = self.inference_stats["total_inferences"]
        self.inference_stats["average_tokens_per_second"] = (
            (self.inference_stats["average_tokens_per_second"] * (total_inferences - 1) + tokens_per_second) 
            / total_inferences
        )
        self.inference_stats["average_memory_usage"] = (
            (self.inference_stats["average_memory_usage"] * (total_inferences - 1) + memory_usage)
            / total_inferences
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.inference_stats.copy()
    
    def reset_stats(self):
        """Reset inference statistics."""
        self.inference_stats = {
            "total_inferences": 0,
            "total_tokens_generated": 0,
            "average_tokens_per_second": 0.0,
            "average_memory_usage": 0.0,
            "error_count": 0
        }
    
    def optimize_for_inference(self):
        """Optimize model for inference."""
        # Set model to eval mode
        self.model.eval()
        
        # Optimize memory if requested
        if self.memory_efficient:
            self.model.optimize_memory()
        
        # Compile model if supported (PyTorch 2.0+)
        if hasattr(torch, 'compile') and hasattr(self.model, 'config'):
            if self.model.config.device.compile_model:
                try:
                    self.model = torch.compile(self.model)
                except Exception as e:
                    print(f"Warning: Model compilation failed: {e}")
    
    def benchmark(self, 
                 num_runs: int = 10,
                 input_length: int = 50,
                 generation_length: int = 50) -> Dict[str, float]:
        """Benchmark inference performance."""
        # Reset stats
        self.reset_stats()
        
        # Create test input
        test_input = torch.randint(1, 1000, (1, input_length))
        
        generation_config = GenerationConfig(
            max_new_tokens=generation_length,
            do_sample=False  # Use greedy for consistent benchmarking
        )
        
        times = []
        memory_usages = []
        
        for _ in range(num_runs):
            start_time = time.time()
            result = self.generate(input_ids=test_input, generation_config=generation_config)
            end_time = time.time()
            
            times.append(end_time - start_time)
            memory_usages.append(result.memory_usage_mb)
        
        return {
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "average_memory_mb": np.mean(memory_usages),
            "average_tokens_per_second": generation_length / np.mean(times)
        }
