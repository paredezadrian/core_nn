#!/usr/bin/env python3
"""
Test suite for CORE-NN core components.

Tests the five main architectural components:
1. Biological Core Memory (BCM)
2. Recursive Temporal Embedding Unit (RTEU)
3. Instruction-Guided Plasticity Module (IGPM)
4. Multi-Level Compression Synthesizer (MLCS)
5. Edge-Efficient Modular Execution Engine
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Import CORE-NN components
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.components.rteu import RecursiveTemporalEmbeddingUnit
from core_nn.components.igpm import InstructionGuidedPlasticityModule
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.components.execution_engine import EdgeEfficientModularExecutionEngine
from core_nn.config.schema import (
    BCMConfig, RTEUConfig, IGPMConfig, MLCSConfig, ExecutionEngineConfig
)


class TestBiologicalCoreMemory:
    """Test suite for Biological Core Memory (BCM)."""
    
    @pytest.fixture
    def bcm_config(self):
        return BCMConfig(
            memory_size=64,
            embedding_dim=128,
            salience_threshold=0.5,
            decay_rate=0.9
        )
    
    @pytest.fixture
    def bcm(self, bcm_config):
        return BiologicalCoreMemory(bcm_config)
    
    def test_bcm_initialization(self, bcm):
        """Test BCM initialization."""
        assert bcm.memory_size == 64
        assert bcm.embedding_dim == 128
        assert len(bcm.memory_slots) == 0
        assert bcm.current_timestamp == 0
    
    def test_bcm_forward_pass(self, bcm):
        """Test BCM forward pass."""
        batch_size = 2
        embedding_dim = 128
        
        input_embedding = torch.randn(batch_size, embedding_dim)
        
        output, info = bcm(input_embedding)
        
        assert output.shape == (batch_size, embedding_dim)
        assert isinstance(info, dict)
        assert "num_stored_memories" in info
        assert "memory_utilization" in info
    
    def test_bcm_explicit_memory(self, bcm):
        """Test explicit memory storage."""
        embedding = torch.randn(128)
        context = {"type": "explicit", "content": "test memory"}
        
        bcm.remember_explicit(embedding, context)
        
        assert len(bcm.memory_slots) == 1
        assert bcm.memory_slots[0].salience == 1.0
        assert bcm.memory_slots[0].context_tags == context
    
    def test_bcm_memory_decay(self, bcm):
        """Test memory decay mechanism."""
        # Add some memories
        for i in range(5):
            embedding = torch.randn(128)
            bcm._store_memory(embedding, 0.8)
        
        initial_count = len(bcm.memory_slots)
        initial_salience = [slot.salience for slot in bcm.memory_slots]
        
        # Apply decay
        bcm._apply_decay()
        
        final_salience = [slot.salience for slot in bcm.memory_slots]
        
        # Check that salience decreased
        for initial, final in zip(initial_salience, final_salience):
            assert final < initial
    
    def test_bcm_memory_stats(self, bcm):
        """Test memory statistics."""
        # Add some memories
        for i in range(3):
            embedding = torch.randn(128)
            bcm._store_memory(embedding, 0.7)
        
        stats = bcm.get_memory_stats()
        
        assert stats["num_memories"] == 3
        assert stats["memory_utilization"] == 3 / 64
        assert "average_salience" in stats


class TestRecursiveTemporalEmbeddingUnit:
    """Test suite for Recursive Temporal Embedding Unit (RTEU)."""
    
    @pytest.fixture
    def rteu_config(self):
        return RTEUConfig(
            num_layers=2,
            embedding_dim=128,
            hidden_dim=256,
            num_capsules=4,
            capsule_dim=32,
            temporal_scales=[1, 4]
        )
    
    @pytest.fixture
    def rteu(self, rteu_config):
        return RecursiveTemporalEmbeddingUnit(rteu_config)
    
    def test_rteu_initialization(self, rteu):
        """Test RTEU initialization."""
        assert rteu.num_layers == 2
        assert len(rteu.layers) == 2
    
    def test_rteu_forward_pass(self, rteu):
        """Test RTEU forward pass."""
        batch_size = 2
        embedding_dim = 128
        
        input_tensor = torch.randn(batch_size, embedding_dim)
        
        output, info = rteu(input_tensor)
        
        assert output.shape == (batch_size, embedding_dim)
        assert isinstance(info, dict)
        assert "layer_activations" in info
        assert len(info["layer_activations"]) == 2
    
    def test_rteu_state_reset(self, rteu):
        """Test RTEU state reset."""
        # Process some data to create states
        input_tensor = torch.randn(1, 128)
        rteu(input_tensor)
        
        # Reset states
        rteu.reset_all_states()
        
        # Check that global state is reset
        assert torch.allclose(rteu.global_state, torch.zeros_like(rteu.global_state))
    
    def test_rteu_temporal_states(self, rteu):
        """Test temporal state retrieval."""
        input_tensor = torch.randn(1, 128)
        rteu(input_tensor)
        
        states = rteu.get_temporal_states()
        
        assert "global_state" in states
        assert len(states) > 1  # Should have capsule states too


class TestInstructionGuidedPlasticityModule:
    """Test suite for Instruction-Guided Plasticity Module (IGPM)."""
    
    @pytest.fixture
    def igpm_config(self):
        return IGPMConfig(
            plastic_slots=16,
            meta_learning_rate=0.01,
            instruction_embedding_dim=64,
            max_episodic_memories=100
        )
    
    @pytest.fixture
    def igpm(self, igpm_config):
        return InstructionGuidedPlasticityModule(igpm_config, vocab_size=1000, embedding_dim=64)
    
    def test_igpm_initialization(self, igpm):
        """Test IGPM initialization."""
        assert len(igpm.slots) == 16
        assert len(igpm.episodic_memories) == 0
    
    def test_igpm_forward_pass(self, igpm):
        """Test IGPM forward pass."""
        batch_size = 2
        embedding_dim = 64
        
        input_tensor = torch.randn(batch_size, embedding_dim)
        instruction = "test instruction"
        
        output, info = igpm(input_tensor, instruction=instruction)
        
        assert output.shape == (batch_size, embedding_dim)
        assert isinstance(info, dict)
        assert "relevant_slots" in info
    
    def test_igpm_explicit_memory(self, igpm):
        """Test explicit memory storage."""
        instruction = "remember this fact"
        context = torch.randn(1, 64)
        
        result = igpm.remember_explicit(instruction, context)
        
        assert result["memory_stored"] == True
        assert len(igpm.episodic_memories) == 1
        assert igpm.episodic_memories[0].instruction == instruction
    
    def test_igpm_memory_recall(self, igpm):
        """Test memory recall by instruction."""
        # Store some memories
        instructions = [
            "the sky is blue",
            "water is wet", 
            "fire is hot"
        ]
        
        for instruction in instructions:
            context = torch.randn(1, 64)
            igpm.remember_explicit(instruction, context)
        
        # Recall memories
        recalled = igpm.recall_by_instruction("sky", top_k=2)
        
        assert len(recalled) > 0
        assert any("sky" in memory.instruction for memory in recalled)


class TestMultiLevelCompressionSynthesizer:
    """Test suite for Multi-Level Compression Synthesizer (MLCS)."""
    
    @pytest.fixture
    def mlcs_config(self):
        return MLCSConfig(
            compression_ratio=0.5,
            num_compression_levels=2,
            latent_dim=64,
            codebook_size=256
        )
    
    @pytest.fixture
    def mlcs(self, mlcs_config):
        return MultiLevelCompressionSynthesizer(mlcs_config, input_dim=128)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_mlcs_initialization(self, mlcs):
        """Test MLCS initialization."""
        assert mlcs.num_levels == 2
        assert len(mlcs.quantizers) == 2
    
    def test_mlcs_compression(self, mlcs):
        """Test knowledge compression."""
        batch_size = 4
        input_dim = 128
        
        knowledge_data = torch.randn(batch_size, input_dim)
        
        kpack = mlcs.compress_knowledge(
            knowledge_data,
            name="test_knowledge",
            description="Test compression"
        )
        
        assert kpack.name == "test_knowledge"
        assert kpack.compressed_size < kpack.original_size
        assert kpack.latent_codes.shape[0] == 2  # num_levels
    
    def test_mlcs_decompression(self, mlcs):
        """Test knowledge decompression."""
        batch_size = 4
        input_dim = 128
        
        # Compress
        knowledge_data = torch.randn(batch_size, input_dim)
        kpack = mlcs.compress_knowledge(knowledge_data)
        
        # Decompress
        reconstructed = mlcs.decompress_knowledge(kpack)
        
        assert reconstructed.shape == knowledge_data.shape
    
    def test_mlcs_save_load(self, mlcs, temp_dir):
        """Test kpack save and load."""
        knowledge_data = torch.randn(2, 128)
        kpack = mlcs.compress_knowledge(knowledge_data, name="test_save")
        
        # Save
        filepath = mlcs.save_kpack(kpack, temp_dir / "test.kpack")
        assert Path(filepath).exists()
        
        # Load
        loaded_kpack = mlcs.load_kpack(filepath)
        
        assert loaded_kpack.name == kpack.name
        assert loaded_kpack.pack_id == kpack.pack_id


class TestEdgeEfficientModularExecutionEngine:
    """Test suite for Edge-Efficient Modular Execution Engine."""
    
    @pytest.fixture
    def engine_config(self):
        return ExecutionEngineConfig(
            max_concurrent_modules=2,
            memory_budget_gb=1,  # Small for testing
            cpu_threads=1
        )
    
    @pytest.fixture
    def engine(self, engine_config):
        return EdgeEfficientModularExecutionEngine(engine_config)
    
    def test_engine_initialization(self, engine):
        """Test execution engine initialization."""
        assert engine.config.max_concurrent_modules == 2
        assert len(engine.module_manager.modules) == 0
    
    def test_engine_component_registration(self, engine):
        """Test component registration."""
        # Create a simple test module
        test_module = torch.nn.Linear(10, 10)
        
        success = engine.register_component("test_module", test_module)
        
        assert success == True
        assert "test_module" in engine.module_manager.modules
        assert "test_module" in engine.module_manager.module_info
    
    def test_engine_system_status(self, engine):
        """Test system status reporting."""
        status = engine.get_system_status()
        
        assert isinstance(status, dict)
        assert "system" in status
        assert "modules" in status
        assert "tasks" in status
    
    def test_engine_memory_optimization(self, engine):
        """Test memory optimization."""
        # Register a test module
        test_module = torch.nn.Linear(100, 100)
        engine.register_component("test_module", test_module)
        
        # Run optimization
        result = engine.optimize_memory()

        assert isinstance(result, dict)
        assert "initial_memory_mb" in result
        assert "final_memory_mb" in result
        assert result["final_memory_mb"] <= result["initial_memory_mb"]


# Integration tests
class TestIntegration:
    """Integration tests for CORE-NN components."""
    
    def test_component_interaction(self):
        """Test interaction between components."""
        # Create configs
        bcm_config = BCMConfig(memory_size=32, embedding_dim=64)
        rteu_config = RTEUConfig(num_layers=1, embedding_dim=64, hidden_dim=128)
        
        # Create components
        bcm = BiologicalCoreMemory(bcm_config)
        rteu = RecursiveTemporalEmbeddingUnit(rteu_config)
        
        # Test data flow
        input_data = torch.randn(1, 64)
        
        # RTEU processing
        rteu_output, _ = rteu(input_data)
        
        # BCM processing
        bcm_output, _ = bcm(rteu_output)
        
        assert bcm_output.shape == input_data.shape


# Performance benchmarks
class TestPerformance:
    """Performance benchmarks for CORE-NN components."""
    
    def test_bcm_performance(self):
        """Benchmark BCM performance."""
        config = BCMConfig(memory_size=256, embedding_dim=512)
        bcm = BiologicalCoreMemory(config)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 512)
            bcm(input_data)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(100):
            input_data = torch.randn(1, 512)
            bcm(input_data)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should process in reasonable time (< 10ms per forward pass)
        assert avg_time < 0.01
    
    def test_rteu_performance(self):
        """Benchmark RTEU performance."""
        config = RTEUConfig(num_layers=2, embedding_dim=256, hidden_dim=512)
        rteu = RecursiveTemporalEmbeddingUnit(config)
        
        # Warm up
        for _ in range(10):
            input_data = torch.randn(1, 256)
            rteu(input_data)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(100):
            input_data = torch.randn(1, 256)
            rteu(input_data)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should process in reasonable time (< 20ms per forward pass)
        assert avg_time < 0.02


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
