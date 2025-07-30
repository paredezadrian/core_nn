"""
Main CORE-NN Model - Integrates all architectural components.

The CoreNNModel combines BCM, RTEU, IGPM, MLCS, and the execution engine
into a unified model for edge-efficient AI processing.

Copyright 2024 Adrian Paredez

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import time

from .components.bcm import BiologicalCoreMemory
from .components.rteu import RecursiveTemporalEmbeddingUnit
from .components.igpm import InstructionGuidedPlasticityModule
from .components.mlcs import MultiLevelCompressionSynthesizer
from .components.execution_engine import EdgeEfficientModularExecutionEngine
from .config.schema import CoreNNConfig
from .tokenization import create_tokenizer_from_config


class CoreNNModel(nn.Module):
    """
    Main CORE-NN Model - Context-Oriented Recurrent Embedding Neural Network.
    
    Integrates all five architectural components:
    1. Biological Core Memory (BCM)
    2. Recursive Temporal Embedding Unit (RTEU) 
    3. Instruction-Guided Plasticity Module (IGPM)
    4. Multi-Level Compression Synthesizer (MLCS)
    5. Edge-Efficient Modular Execution Engine
    """
    
    def __init__(self, config: CoreNNConfig, vocab_size: int = 50000):
        super().__init__()
        self.config = config

        # Initialize tokenizer
        self.tokenizer = create_tokenizer_from_config(config.tokenizer)
        # Use a larger vocab size to account for character fallback tokens (up to 65536)
        base_vocab_size = self.tokenizer.get_vocab_size()
        self.vocab_size = max(base_vocab_size, 65536)  # Ensure we can handle character fallback

        # Core architectural components
        self.bcm = BiologicalCoreMemory(config.bcm)
        self.rteu = RecursiveTemporalEmbeddingUnit(config.rteu)
        self.igpm = InstructionGuidedPlasticityModule(config.igpm, self.vocab_size, config.rteu.embedding_dim)
        self.mlcs = MultiLevelCompressionSynthesizer(config.mlcs, config.rteu.embedding_dim)
        self.execution_engine = EdgeEfficientModularExecutionEngine(config.execution_engine)
        
        # Input/Output layers
        self.token_embedding = nn.Embedding(self.vocab_size, config.rteu.embedding_dim)
        self.position_embedding = nn.Embedding(config.inference.max_sequence_length, config.rteu.embedding_dim)
        self.output_projection = nn.Linear(config.rteu.embedding_dim, self.vocab_size)
        
        # Layer normalization
        self.input_norm = nn.LayerNorm(config.rteu.embedding_dim)
        self.output_norm = nn.LayerNorm(config.rteu.embedding_dim)
        
        # Register components with execution engine
        self._register_components()
        
        # Model state
        self.current_position = 0
        self.session_active = False
        
    def _register_components(self):
        """Register components with the execution engine."""
        self.execution_engine.register_component("bcm", self.bcm, priority=3)
        self.execution_engine.register_component("rteu", self.rteu, priority=2)
        self.execution_engine.register_component("igpm", self.igpm, priority=1)
        self.execution_engine.register_component("mlcs", self.mlcs, priority=0)
    
    def forward(self, 
                input_ids: torch.Tensor,
                instruction: Optional[str] = None,
                instruction_tokens: Optional[torch.Tensor] = None,
                reset_state: bool = False) -> Dict[str, Any]:
        """
        Forward pass through CORE-NN model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            instruction: Optional natural language instruction
            instruction_tokens: Optional tokenized instruction
            reset_state: Whether to reset temporal states
            
        Returns:
            Dictionary containing outputs and component information
        """
        batch_size, seq_len = input_ids.shape
        
        if reset_state:
            self.reset_states()
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(self.current_position, 
                               self.current_position + seq_len, 
                               device=input_ids.device)
        position_embeds = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine embeddings
        embeddings = self.input_norm(token_embeds + position_embeds)
        
        # Process each token in sequence
        outputs = []
        component_info = {
            "bcm_info": [],
            "rteu_info": [],
            "igpm_info": [],
            "mlcs_info": []
        }
        
        for t in range(seq_len):
            current_embed = embeddings[:, t, :]  # [batch_size, embedding_dim]
            
            # 1. Process through RTEU (temporal processing)
            rteu_output, rteu_info = self.rteu(current_embed, reset_state=(t == 0 and reset_state))
            component_info["rteu_info"].append(rteu_info)
            
            # 2. Process through BCM (memory storage and retrieval)
            bcm_output, bcm_info = self.bcm(rteu_output, query_embedding=rteu_output)
            component_info["bcm_info"].append(bcm_info)
            
            # 3. Process through IGPM (instruction-guided plasticity)
            igpm_output, igpm_info = self.igpm(
                bcm_output, 
                instruction=instruction,
                instruction_tokens=instruction_tokens
            )
            component_info["igpm_info"].append(igpm_info)
            
            # 4. Final processing and output projection
            final_output = self.output_norm(igpm_output)
            logits = self.output_projection(final_output)
            
            outputs.append(logits)
        
        # Stack outputs
        output_logits = torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]
        
        # Update position counter
        self.current_position += seq_len
        
        # Prepare comprehensive output
        result = {
            "logits": output_logits,
            "last_hidden_state": final_output,
            "component_info": component_info,
            "model_info": {
                "current_position": self.current_position,
                "session_active": self.session_active,
                "memory_stats": self.get_memory_stats(),
                "system_status": self.execution_engine.get_system_status()
            }
        }
        
        return result
    
    def generate(self, 
                input_ids: torch.Tensor,
                max_new_tokens: int = 50,
                temperature: float = 0.7,
                top_k: int = 50,
                top_p: float = 0.9,
                instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate text using the CORE-NN model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            instruction: Optional instruction for generation
            
        Returns:
            Dictionary containing generated tokens and metadata
        """
        self.eval()
        generated_tokens = []
        generation_info = []
        
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    current_ids[:, -1:],  # Only process last token
                    instruction=instruction,
                    reset_state=(step == 0)
                )
                
                # Get logits for last position
                logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    # Ensure top_k doesn't exceed vocabulary size
                    actual_top_k = min(top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(logits, actual_top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Store generation info
                generation_info.append({
                    "step": step,
                    "token": next_token.item(),
                    "probability": probs[0, next_token].item(),
                    "component_info": outputs["component_info"]
                })
                
                # Check for early stopping (e.g., EOS token)
                if next_token.item() == 0:  # Assuming 0 is EOS token
                    break
        
        return {
            "generated_ids": current_ids,
            "generated_tokens": generated_tokens,
            "generation_info": generation_info,
            "total_tokens": len(generated_tokens)
        }
    
    def remember(self, instruction: str, context: Optional[torch.Tensor] = None):
        """Explicitly remember instruction and context."""
        if context is None:
            # Create dummy context
            context = torch.randn(1, self.config.rteu.embedding_dim)
        
        # Store in BCM
        self.bcm.remember_explicit(context, {"instruction": instruction, "explicit": True})
        
        # Store in IGPM
        return self.igpm.remember_explicit(instruction, context)
    
    def recall(self, query: str, top_k: int = 5) -> List[Any]:
        """Recall memories based on query."""
        # Recall from IGPM
        episodic_memories = self.igpm.recall_by_instruction(query, top_k)
        
        # Recall from BCM
        bcm_memories = self.bcm.recall_by_context({"instruction": query}, top_k)
        
        return {
            "episodic_memories": episodic_memories,
            "bcm_memories": bcm_memories
        }
    
    def forget(self, query: str) -> Dict[str, Any]:
        """Forget memories related to query."""
        # This is a simplified implementation
        # In practice, would need more sophisticated forgetting mechanisms
        
        initial_bcm_count = len(self.bcm.memory_slots)
        initial_igpm_count = len(self.igpm.episodic_memories)
        
        # Remove matching memories (simplified)
        self.bcm.memory_slots = [
            slot for slot in self.bcm.memory_slots
            if not (slot.context_tags and 
                   "instruction" in slot.context_tags and 
                   query.lower() in slot.context_tags["instruction"].lower())
        ]
        
        # Remove from IGPM episodic memories
        self.igpm.episodic_memories = [
            memory for memory in self.igpm.episodic_memories
            if query.lower() not in memory.instruction.lower()
        ]
        
        final_bcm_count = len(self.bcm.memory_slots)
        final_igpm_count = len(self.igpm.episodic_memories)
        
        return {
            "bcm_memories_removed": initial_bcm_count - final_bcm_count,
            "igpm_memories_removed": initial_igpm_count - final_igpm_count
        }
    
    def reset_states(self):
        """Reset all temporal states."""
        self.rteu.reset_all_states()
        self.current_position = 0
        
    def start_session(self):
        """Start a new session."""
        self.session_active = True
        self.reset_states()
        
    def end_session(self):
        """End current session."""
        self.session_active = False
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "bcm_stats": self.bcm.get_memory_stats(),
            "igpm_stats": self.igpm.get_plasticity_stats(),
            "mlcs_stats": self.mlcs.get_compression_stats(),
            "rteu_states": len(self.rteu.get_temporal_states())
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage across all components."""
        # Optimize execution engine memory
        engine_optimization = self.execution_engine.optimize_memory()
        
        # Apply memory decay in BCM
        initial_bcm_memories = len(self.bcm.memory_slots)
        self.bcm._apply_decay()
        final_bcm_memories = len(self.bcm.memory_slots)
        
        # Manage MLCS memory
        self.mlcs.manage_memory()
        
        return {
            "execution_engine": engine_optimization,
            "bcm_memories_decayed": initial_bcm_memories - final_bcm_memories,
            "mlcs_optimization": "completed",
            "total_memory_stats": self.get_memory_stats()
        }
