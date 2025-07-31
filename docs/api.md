# CORE-NN API Reference

This document provides comprehensive API documentation for CORE-NN (Context-Oriented Recurrent Embedding Neural Network).

**Author:** Adrian Paredez ([@paredezadrian](https://github.com/paredezadrian))
**Repository:** https://github.com/paredezadrian/core_nn.git
**Version:** 0.3.0

## Table of Contents

- [Core Model API](#core-model-api)
- [Component APIs](#component-apis)
- [Configuration API](#configuration-api)
- [CLI Commands](#cli-commands)
- [Memory Commands](#memory-commands)
- [Inference API](#inference-api)
- [Utilities](#utilities)

## Core Model API

### CoreNNModel

The main model class that integrates all CORE-NN components.

```python
from core_nn import CoreNNModel, ConfigManager

# Initialize model with optimized configuration (recommended)
config_manager = ConfigManager()
config = config_manager.load_config('configs/laptop_optimized_flexible_sequences.yaml')
model = CoreNNModel(config)
```

### Optimized Model Variants

#### LongContextCoreNNModel

Extended model for long-context processing with support for 8000+ tokens.

```python
from optimization.long_context_fix import LongContextCoreNNModel

# Initialize long-context model
long_context_model = LongContextCoreNNModel(
    config, 
    vocab_size=50000, 
    max_sequence_length=8192
)

# Process long sequences
result = long_context_model(input_ids)  # Supports 8000+ tokens
```

#### MemoryOptimizedCoreNNModel

Memory-optimized model with adaptive chunk sizing and memory management.

```python
from optimization.long_context_optimization import MemoryOptimizedCoreNNModel

# Initialize memory-optimized model
memory_model = MemoryOptimizedCoreNNModel(
    config, 
    vocab_size=50000, 
    max_sequence_length=8192,
    memory_limit_gb=10.0
)

# Process with memory constraints
result = memory_model(input_ids)  # Automatic memory management
```

#### Methods

##### `__init__(config: CoreNNConfig)`
Initialize the CORE-NN model with configuration.

**Parameters:**
- `config`: CoreNNConfig object containing all component configurations

##### `forward(input_ids, instruction=None, instruction_tokens=None, reset_state=False)`
Forward pass through the model.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `instruction` (str, optional): Natural language instruction
- `instruction_tokens` (torch.Tensor, optional): Tokenized instruction
- `reset_state` (bool): Whether to reset temporal states

**Returns:**
- `dict`: Contains 'logits', 'last_hidden_state', 'component_info', 'model_info'

##### `generate(input_ids, max_new_tokens=50, temperature=0.7, top_k=50, top_p=0.9, instruction=None)`
Generate text using the model.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `max_new_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature
- `top_k` (int): Top-k sampling parameter
- `top_p` (float): Top-p (nucleus) sampling parameter
- `instruction` (str, optional): Generation instruction

**Returns:**
- `dict`: Contains 'generated_tokens', 'generated_text', 'generation_info'

##### `remember(instruction: str, context: torch.Tensor = None)`
Explicitly remember instruction and context.

**Parameters:**
- `instruction` (str): Instruction to remember
- `context` (torch.Tensor, optional): Context tensor

**Returns:**
- `dict`: Memory storage result with 'memory_stored' boolean

##### `recall(query: str, top_k: int = 5)`
Recall memories based on query.

**Parameters:**
- `query` (str): Query string
- `top_k` (int): Number of memories to recall

**Returns:**
- `dict`: Dictionary with 'episodic_memories' and 'bcm_memories' keys

##### `forget(query: str)`
Forget memories related to query.

**Parameters:**
- `query` (str): Query for memories to forget

**Returns:**
- `dict`: Forgetting result with count of forgotten memories

##### `start_session()`
Start a new session.

**Note:** Session names are managed by the SessionManager, not the model directly.

##### `end_session()`
End the current session and save state.

##### `reset_states()`
Reset all temporal states in the model.

##### `get_memory_stats()`
Get comprehensive memory statistics.

**Returns:**
- `dict`: Memory usage statistics for all components

## Component APIs

### BiologicalCoreMemory (BCM)

Hippocampus-inspired temporal memory system.

```python
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.config.schema import BCMConfig

config = BCMConfig(memory_size=512, salience_threshold=0.7)
bcm = BiologicalCoreMemory(config)
```

#### Methods

##### `forward(input_embedding, query_embedding=None)`
Process input through BCM.

**Parameters:**
- `input_embedding` (torch.Tensor): Current input [batch_size, embedding_dim]
- `query_embedding` (torch.Tensor, optional): Query for retrieval

**Returns:**
- `tuple`: (retrieved_memory, memory_info)

##### `remember_explicit(embedding, metadata=None)`
Explicitly store a memory.

**Parameters:**
- `embedding` (torch.Tensor): Embedding to store
- `metadata` (dict, optional): Additional metadata

### RecursiveTemporalEmbeddingUnit (RTEU)

Multi-timescale temporal processing unit.

```python
from core_nn.components.rteu import RecursiveTemporalEmbeddingUnit
from core_nn.config.schema import RTEUConfig

config = RTEUConfig(num_layers=4, temporal_scales=[1, 4, 16, 64])
rteu = RecursiveTemporalEmbeddingUnit(config)
```

#### Methods

##### `forward(x)`
Process input through RTEU.

**Parameters:**
- `x` (torch.Tensor): Input tensor [batch_size, embedding_dim]

**Returns:**
- `tuple`: (output, info_dict)

##### `reset_all_states()`
Reset all temporal states.

### InstructionGuidedPlasticityModule (IGPM)

Instruction-guided adaptation without global updates.

```python
from core_nn.components.igpm import InstructionGuidedPlasticityModule
from core_nn.config.schema import IGPMConfig

config = IGPMConfig(plastic_slots=64, meta_learning_rate=0.001)
igpm = InstructionGuidedPlasticityModule(config, vocab_size=50000, embedding_dim=768)
```

#### Methods

##### `forward(x, instruction=None, instruction_tokens=None)`
Apply plastic transformation.

**Parameters:**
- `x` (torch.Tensor): Input tensor [batch_size, embedding_dim]
- `instruction` (str, optional): Natural language instruction
- `instruction_tokens` (torch.Tensor, optional): Tokenized instruction

**Returns:**
- `tuple`: (output, info_dict)

##### `remember_explicit(instruction, context)`
Store explicit episodic memory.

**Parameters:**
- `instruction` (str): Instruction text
- `context` (torch.Tensor): Context embedding

**Returns:**
- `dict`: Storage result

##### `recall_by_instruction(instruction, top_k=5)`
Recall memories by instruction similarity.

**Parameters:**
- `instruction` (str): Query instruction
- `top_k` (int): Number of memories to recall

**Returns:**
- `list`: List of EpisodicMemory objects

### MultiLevelCompressionSynthesizer (MLCS)

Knowledge compression and .kpack management.

```python
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.config.schema import MLCSConfig

config = MLCSConfig(latent_dim=256, num_compression_levels=4)
mlcs = MultiLevelCompressionSynthesizer(config, input_dim=768)
```

#### Methods

##### `compress_knowledge(knowledge_data, compression_level=-1, name="unnamed", description="")`
Compress knowledge into a KnowledgePack.

**Parameters:**
- `knowledge_data` (torch.Tensor): Data to compress
- `compression_level` (int): Target compression level (-1 for maximum)
- `name` (str): Name for the knowledge pack
- `description` (str): Description

**Returns:**
- `KnowledgePack`: Compressed knowledge pack

##### `decompress_knowledge(kpack, from_level=0)`
Decompress knowledge from a pack.

**Parameters:**
- `kpack` (KnowledgePack): Knowledge pack to decompress
- `from_level` (int): Decompression starting level

**Returns:**
- `torch.Tensor`: Decompressed knowledge

##### `load_knowledge_capsule(kpack_path)`
Load a .kpack file.

**Parameters:**
- `kpack_path` (str): Path to .kpack file

**Returns:**
- `str`: Pack ID for loaded knowledge

##### `unload_knowledge_capsule(pack_id)`
Unload a knowledge pack from memory.

**Parameters:**
- `pack_id` (str): ID of pack to unload

## Configuration API

### ConfigManager

Manages configuration loading and validation.

```python
from core_nn import ConfigManager

config_manager = ConfigManager()
```

#### Methods

##### `load_config(config_path)`
Load configuration from file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `CoreNNConfig`: Loaded configuration object

##### `save_config(config, config_path)`
Save configuration to file.

**Parameters:**
- `config` (CoreNNConfig): Configuration to save
- `config_path` (str): Output file path

##### `validate_config(config)`
Validate configuration object.

**Parameters:**
- `config` (CoreNNConfig): Configuration to validate

**Returns:**
- `bool`: True if valid

##### `get_available_configs()`
Get list of available configuration templates.

**Returns:**
- `list`: List of available config names

### Configuration Classes

All configuration classes are dataclasses with sensible defaults:

- `CoreNNConfig`: Main configuration container
- `BCMConfig`: Biological Core Memory configuration
- `RTEUConfig`: Recursive Temporal Embedding Unit configuration
- `IGPMConfig`: Instruction-Guided Plasticity Module configuration
- `MLCSConfig`: Multi-Level Compression Synthesizer configuration
- `ExecutionEngineConfig`: Execution engine configuration
- `DeviceConfig`: Device and hardware configuration
- `InferenceConfig`: Inference parameters
- `MemoryConfig`: Memory management settings
- `LoggingConfig`: Logging configuration
- `SessionConfig`: Session management settings

## CLI Commands

### Initialize Project

```bash
core-nn init [OPTIONS]
```

**Options:**
- `--config-template, -t`: Configuration template (default, edge_device, minimal)
- `--output-dir, -o`: Output directory
- `--force, -f`: Force overwrite existing files

### Interactive Chat

```bash
core-nn chat [OPTIONS]
```

**Options:**
- `--model-path, -m`: Path to saved model
- `--session-name, -s`: Session name
- `--max-tokens`: Maximum tokens per response
- `--temperature`: Sampling temperature

### Validate Configuration

```bash
core-nn validate [OPTIONS]
```

**Options:**
- `--config-file, -c`: Configuration file to validate

### Show Information

```bash
core-nn info [OPTIONS]
```

Shows system information, available configurations, and model status.

## Memory Commands

Available in interactive chat mode:

### `/remember <text>`
Explicitly remember information.

**Example:**
```
/remember The capital of France is Paris
```

### `/recall <query>`
Recall memories related to query.

**Example:**
```
/recall capital of France
```

### `/forget <query>`
Forget memories related to query.

**Example:**
```
/forget outdated information
```

### `/stats`
Show memory and system statistics.

## Inference API

### InferenceEngine

High-level inference interface.

```python
from core_nn.inference import InferenceEngine

engine = InferenceEngine(model, config)
```

#### Methods

##### `generate_text(prompt, max_tokens=50, **kwargs)`
Generate text from prompt.

##### `chat_completion(messages, **kwargs)`
Complete a chat conversation.

### SessionManager

Manages conversation sessions.

```python
from core_nn.inference import SessionManager

session_manager = SessionManager(config.session)
```

#### Methods

##### `create_session(name=None)`
Create a new session.

##### `load_session(session_id)`
Load an existing session.

##### `save_session(session_id)`
Save current session state.

## Utilities

### Device Management

```python
from core_nn.utils import get_optimal_device

device = get_optimal_device()  # Returns best available device
```

### Profiling

```python
from core_nn.utils import profile_memory, profile_compute

# Profile memory usage
with profile_memory() as profiler:
    result = model(input_ids)
print(profiler.get_stats())

# Profile compute time
with profile_compute() as profiler:
    result = model(input_ids)
print(profiler.get_stats())
```

### Logging

```python
from core_nn.utils import setup_logging

setup_logging(level="INFO", log_file="core_nn.log")
```

## Error Handling

CORE-NN uses custom exceptions for better error handling:

- `CoreNNConfigError`: Configuration-related errors
- `CoreNNMemoryError`: Memory management errors
- `CoreNNInferenceError`: Inference-related errors
- `CoreNNTokenizationError`: Tokenization errors

## Complete Usage Examples

### Basic Text Generation

```python
import torch
from core_nn import CoreNNModel, ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config('configs/default.yaml')

# Create model
model = CoreNNModel(config)
model.eval()

# Generate text
input_text = "The future of AI is"
input_ids = model.tokenizer.encode(input_text, return_tensors='pt')

with torch.no_grad():
    result = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9
    )

print(f"Generated: {result['generated_text']}")
```

### Memory Operations

```python
# Start a session
model.start_session("my_session")

# Remember information
model.remember("The capital of France is Paris")
model.remember("Python is a programming language")

# Recall information
memories = model.recall("capital")
for memory in memories:
    print(f"Recalled: {memory.instruction}")

# Forget information
result = model.forget("outdated")
print(f"Forgot {result['forgotten_count']} memories")

# End session
model.end_session()
```

### Custom Configuration

```python
from core_nn.config.schema import *

# Create custom configuration
config = CoreNNConfig(
    model=ModelConfig(name="my-custom-model"),
    bcm=BCMConfig(
        memory_size=256,
        salience_threshold=0.5
    ),
    rteu=RTEUConfig(
        num_layers=2,
        embedding_dim=512,
        temporal_scales=[1, 4, 8]
    ),
    igpm=IGPMConfig(
        plastic_slots=32,
        meta_learning_rate=0.01
    ),
    inference=InferenceConfig(
        max_sequence_length=1024,
        temperature=0.8
    )
)

# Use custom configuration
model = CoreNNModel(config)
```

### Knowledge Pack Operations

```python
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.config.schema import MLCSConfig

# Create MLCS
config = MLCSConfig(latent_dim=256, num_compression_levels=3)
mlcs = MultiLevelCompressionSynthesizer(config, input_dim=768)

# Compress knowledge
knowledge_data = torch.randn(10, 768)  # Some knowledge to compress
kpack = mlcs.compress_knowledge(
    knowledge_data,
    name="my_knowledge",
    description="Custom knowledge pack"
)

# Save knowledge pack
from core_nn.memory.kpack import save_kpack
save_kpack(kpack, "my_knowledge.kpack")

# Load and use knowledge pack
from core_nn.memory.kpack import load_kpack
loaded_kpack = load_kpack("my_knowledge.kpack")
reconstructed = mlcs.decompress_knowledge(loaded_kpack)
```

### Batch Processing

```python
# Process multiple inputs
batch_inputs = [
    "What is machine learning?",
    "Explain neural networks",
    "How does CORE-NN work?"
]

results = []
for input_text in batch_inputs:
    input_ids = model.tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        result = model.generate(input_ids, max_new_tokens=30)

    results.append(result['generated_text'])

for i, result in enumerate(results):
    print(f"Input {i+1}: {batch_inputs[i]}")
    print(f"Output {i+1}: {result}\n")
```

### Component-Level Usage

```python
# Use individual components
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.config.schema import BCMConfig

# Create and use BCM
bcm_config = BCMConfig(memory_size=128, salience_threshold=0.6)
bcm = BiologicalCoreMemory(bcm_config)

# Process input
input_embedding = torch.randn(2, 768)
output, info = bcm(input_embedding)

print(f"Stored memories: {info['num_stored_memories']}")
print(f"Average salience: {info['average_salience']:.3f}")
```

### Error Handling

```python
from core_nn.exceptions import CoreNNConfigError, CoreNNMemoryError

try:
    # Load configuration
    config = config_manager.load_config('invalid_config.yaml')
except CoreNNConfigError as e:
    print(f"Configuration error: {e}")
    # Use default configuration
    config = config_manager.get_default_config()

try:
    # Memory operation
    model.remember("Some information")
except CoreNNMemoryError as e:
    print(f"Memory error: {e}")
    # Handle memory full condition
    model.forget("old information")
```

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic model usage
- `memory_operations.py`: Memory command examples
- `custom_configuration.py`: Custom configuration setup
- `batch_processing.py`: Batch inference examples
- `component_usage.py`: Individual component usage
- `knowledge_pack_demo.py`: Knowledge pack operations
