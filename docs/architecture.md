# CORE-NN Architecture Guide

## Overview

CORE-NN (Context-Oriented Recurrent Embedding Neural Network) is a novel AI architecture designed specifically for edge devices. Unlike traditional transformer-based models, CORE-NN uses a modular, memory-efficient approach that combines biological principles with modern deep learning techniques.

## Core Principles

### 1. Biological Inspiration
- **Hippocampus-like Memory**: The BCM mimics hippocampal memory formation and consolidation
- **Working Memory**: Temporal processing similar to human working memory
- **Selective Attention**: Salience-based filtering of information

### 2. Edge Optimization
- **Memory Efficiency**: Dynamic memory management and compression
- **Modular Execution**: Asynchronous processing with component offloading
- **CPU-First Design**: Optimized for CPU execution with optional GPU acceleration

### 3. Adaptive Learning
- **Meta-Learning**: Fast adaptation without global weight updates
- **Instruction-Guided**: Natural language instructions guide model behavior
- **Episodic Memory**: Explicit memory storage and retrieval

## Architecture Components

### 1. Biological Core Memory (BCM)

The BCM serves as the central memory system, inspired by the hippocampus.

**Key Features:**
- Fixed-size sliding window of embeddings
- Salience-based memory retention
- Temporal decay mechanism
- Multi-head attention for retrieval

**Configuration:**
```yaml
bcm:
  memory_size: 512        # Number of memory slots
  embedding_dim: 768      # Dimension of embeddings
  salience_threshold: 0.7 # Threshold for memory retention
  decay_rate: 0.95        # Memory decay rate
  attention_heads: 8      # Number of attention heads
```

**Memory Storage Process:**
1. Input embeddings are evaluated for salience
2. High-salience embeddings are stored in memory slots
3. Memory undergoes temporal decay over time
4. Least salient memories are replaced when capacity is reached

### 2. Recursive Temporal Embedding Unit (RTEU)

The RTEU replaces traditional self-attention with temporal routing mechanisms.

**Key Features:**
- Multi-timescale temporal capsules
- Routing-by-agreement mechanism
- Recurrent state management
- Efficient CPU processing

**Configuration:**
```yaml
rteu:
  num_layers: 4
  embedding_dim: 768
  hidden_dim: 2048
  num_capsules: 16
  capsule_dim: 48
  routing_iterations: 3
  temporal_scales: [1, 4, 16, 64]  # Fast to slow timescales
```

**Processing Flow:**
1. Input is processed by temporal capsules at different timescales
2. Capsule outputs are combined using routing-by-agreement
3. Feed-forward network processes the routed output
4. Residual connections and layer normalization are applied

### 3. Instruction-Guided Plasticity Module (IGPM) ✨ OPTIMIZED v0.2.2

The IGPM enables sophisticated fast adaptation through biologically-inspired plasticity mechanisms.

**🎉 MAJOR BREAKTHROUGH**: Complete transformation from non-functional (0.0000 change magnitude) to highly responsive system (5.5+ change magnitude).

**Enhanced Key Features:**
- **Gradient-Based Plasticity**: Momentum-based gradient accumulation with adaptive learning rates
- **MAML-Style Meta-Learning**: Multi-step inner loop adaptation (3 steps) with context encoding
- **Context-Dependent Plasticity**: Different plasticity rules for memory/attention/suppression instructions
- **Neuromodulation-Inspired Plasticity**: 4 neurotransmitter systems (dopamine, acetylcholine, norepinephrine, serotonin)
- **Enhanced Architecture**: LayerNorm, Tanh activations, deeper networks, robust error handling
- **Episodic Memory Integration**: Improved storage and retrieval with plasticity feedback

**Configuration:**
```yaml
igpm:
  plastic_slots: 64
  meta_learning_rate: 0.1  # Enhanced learning rate
  fast_weight_decay: 0.99
  instruction_embedding_dim: 256
  max_episodic_memories: 1000
  plasticity_threshold: 0.001  # Lowered for better activation
```

**Enhanced Adaptation Process:**
1. Instructions are encoded with improved embedding networks
2. Context-dependent plasticity rules are applied based on instruction type
3. Gradient-based fast weight updates with momentum and adaptive learning
4. MAML-style multi-step adaptation for faster convergence
5. Neuromodulation factors adjust plasticity strength dynamically
6. Episodic memories are stored with enhanced plasticity feedback

### 4. Multi-Level Compression Synthesizer (MLCS)

The MLCS manages knowledge compression and storage in .kpack files.

**Key Features:**
- Hierarchical compression levels
- Vector quantization
- Dynamic loading/unloading
- Lightweight knowledge packs

**Configuration:**
```yaml
mlcs:
  compression_ratio: 0.1
  num_compression_levels: 4
  latent_dim: 256
  codebook_size: 8192
  kpack_max_size_mb: 50
```

**Compression Pipeline:**
1. Knowledge is encoded through hierarchical encoder
2. Vector quantization compresses representations
3. Compressed data is stored in .kpack files
4. Knowledge packs can be dynamically loaded as needed

### 5. Edge-Efficient Modular Execution Engine

The execution engine manages component lifecycle and resource allocation.

**Key Features:**
- Asynchronous module execution
- Dynamic component offloading
- Memory budget management
- Priority-based scheduling

**Configuration:**
```yaml
execution_engine:
  max_concurrent_modules: 4
  memory_budget_gb: 12
  cpu_threads: -1  # Auto-detect
  offload_threshold: 0.8
  async_execution: true
```

**Resource Management:**
1. Components are registered with priority levels
2. Memory usage is continuously monitored
3. Unused components are offloaded when memory is tight
4. Tasks are scheduled based on priority and resource availability

## Data Flow

### Forward Pass
1. **Input Processing**: Tokens are embedded and position-encoded
2. **RTEU Processing**: Temporal processing with multi-timescale capsules
3. **BCM Integration**: Memory storage and retrieval operations
4. **IGPM Adaptation**: Instruction-guided plasticity application
5. **Output Generation**: Final processing and token prediction

### Memory Operations
1. **Storage**: High-salience information is stored in BCM and IGPM
2. **Retrieval**: Query-based memory retrieval using attention mechanisms
3. **Consolidation**: Periodic memory consolidation and compression
4. **Forgetting**: Selective memory removal based on decay and relevance

## Performance Characteristics

### Memory Usage
| Component | Typical RAM | Disk Storage | Notes |
|-----------|-------------|--------------|-------|
| BCM (512 slots) | ~1 GB | None | In-memory only |
| RTEU (4 layers) | ~5 GB | None | Model weights |
| IGPM | ~2 GB | Optional | Session-based |
| MLCS | ~2-3 GB | ~50 MB/kpack | Dynamic loading |
| **Total** | **~10-12 GB** | **<1 GB** | With 10 kpacks |

### Computational Complexity
- **BCM**: O(n × d × h) where n=batch_size, d=embedding_dim, h=attention_heads
- **RTEU**: O(n × d × l × c) where l=num_layers, c=num_capsules
- **IGPM**: O(n × d × s) where s=plastic_slots
- **MLCS**: O(n × d × log(k)) where k=codebook_size

## Deployment Configurations

### Edge Device (Laptop/Mobile)
```yaml
# Optimized for 8GB RAM, CPU-only
bcm:
  memory_size: 256
  embedding_dim: 512
rteu:
  num_layers: 2
  hidden_dim: 1024
execution_engine:
  memory_budget_gb: 6
  max_concurrent_modules: 2
```

### Server Deployment
```yaml
# Optimized for high-performance servers
bcm:
  memory_size: 1024
  embedding_dim: 1024
rteu:
  num_layers: 6
  hidden_dim: 4096
execution_engine:
  memory_budget_gb: 32
  max_concurrent_modules: 8
```

### Mobile Device
```yaml
# Optimized for smartphones/tablets
bcm:
  memory_size: 128
  embedding_dim: 256
rteu:
  num_layers: 1
  hidden_dim: 512
execution_engine:
  memory_budget_gb: 3
  max_concurrent_modules: 1
```

## Comparison with Transformers

| Aspect | CORE-NN | Transformers |
|--------|---------|--------------|
| **Attention** | Temporal routing | Self-attention |
| **Memory** | Explicit biological memory | Implicit in weights |
| **Adaptation** | Fast meta-learning | Fine-tuning required |
| **Edge Efficiency** | Optimized for edge | Resource intensive |
| **Modularity** | Highly modular | Monolithic |
| **State Management** | Explicit temporal states | Stateless |

## Future Directions

### Planned Enhancements
1. **Advanced Compression**: Improved compression algorithms for .kpack files
2. **Distributed Execution**: Multi-device execution capabilities
3. **Online Learning**: Continuous learning from user interactions
4. **Specialized Modules**: Domain-specific component variants

### Research Areas
1. **Neuromorphic Integration**: Hardware acceleration using neuromorphic chips
2. **Federated Learning**: Distributed training across edge devices
3. **Causal Reasoning**: Enhanced reasoning capabilities
4. **Multimodal Processing**: Vision and audio integration

## References

1. Biological memory systems and hippocampal function
2. Capsule networks and routing algorithms
3. Meta-learning and few-shot adaptation
4. Edge computing and resource optimization
5. Vector quantization and compression techniques
