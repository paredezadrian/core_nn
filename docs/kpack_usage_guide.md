# .kpack Knowledge Capsule Usage Guide

## Overview

The `.kpack` Knowledge Capsule system provides a robust, compressed format for storing and transferring learned knowledge representations in CORE-NN. Knowledge capsules can be loaded into the MLCS (Multi-Level Compression Synthesizer) module for dynamic knowledge injection and exported for sharing or archival.

## Quick Start

### Basic Usage

```python
from core_nn.memory.kpack import create_capsule, save_kpack, load_kpack
import numpy as np

# Create a knowledge capsule
capsule = create_capsule(
    name="My Knowledge",
    topic="machine_learning",
    description="Custom ML knowledge capsule",
    embeddings={
        "concepts": np.random.randn(100, 768).astype(np.float32)
    }
)

# Save to file
save_kpack(capsule, "my_knowledge.kpack")

# Load from file
loaded_capsule = load_kpack("my_knowledge.kpack")
print(f"Loaded: {loaded_capsule.metadata.name}")
```

### MLCS Integration

```python
from core_nn.components.mlcs import MultiLevelCompressionSynthesizer
from core_nn.config.schema import MLCSConfig

# Create MLCS instance
config = MLCSConfig(latent_dim=256, num_compression_levels=3)
mlcs = MultiLevelCompressionSynthesizer(config)

# Load knowledge capsule into MLCS
pack_id = mlcs.load_knowledge_capsule("examples/kpacks/ml_fundamentals.kpack")
print(f"Loaded capsule with ID: {pack_id}")

# Compress new knowledge and save as capsule
knowledge_data = torch.randn(100, 768)
compressed_pack = mlcs.compress_knowledge(knowledge_data, name="New Knowledge")
mlcs.save_knowledge_capsule(compressed_pack.pack_id, "new_knowledge.kpack")

# Export all learned knowledge
mlcs.export_learned_knowledge(
    topic="combined_knowledge",
    path="exported_knowledge.kpack",
    name="All Learned Knowledge"
)
```

## Key Features

### **Efficient Compression**
- ZSTD and GZIP compression for different data types
- Quantization support for embeddings
- Sparse tensor format support
- Typical compression ratios: 0.005-0.1 (99.5%-90% size reduction)

### **Data Integrity**
- SHA-256 checksums for corruption detection
- Format version validation
- Schema validation for metadata
- Size verification and bounds checking

### **Flexible Structure**
- Multi-level embeddings with different dimensions
- Knowledge sketches (attention, gradient, activation patterns)
- Rich metadata with provenance tracking
- Extensible compression data section

### **MLCS Integration**
- Seamless loading into MLCS for knowledge injection
- Export MLCS-compressed knowledge as capsules
- Memory management and automatic cleanup
- Batch processing of multiple capsules

## File Format Structure

```
.kpack File Layout:
┌─────────────────────────────────────┐
│ Header (32 bytes)                   │  ← Magic number, version, sizes
├─────────────────────────────────────┤
│ Metadata Section (compressed JSON)  │  ← Name, description, provenance
├─────────────────────────────────────┤
│ Embeddings Section (compressed)     │  ← Multi-level embeddings
├─────────────────────────────────────┤
│ Sketches Section (compressed)       │  ← Knowledge sketches
├─────────────────────────────────────┤
│ Compression Data (compressed JSON)  │  ← MLCS-specific data
├─────────────────────────────────────┤
│ Checksum (32 bytes)                 │  ← SHA-256 integrity check
└─────────────────────────────────────┘
```

## Example Capsules

The system includes several example capsules demonstrating different use cases:

### **ML Fundamentals** (`ml_fundamentals.kpack`)
- Core machine learning concepts
- Supervised/unsupervised learning embeddings
- Neural network and optimization knowledge
- Size: ~1.4MB → ~7KB (99.5% compression)

### **NLP Knowledge** (`nlp_knowledge.kpack`)
- Natural language processing representations
- Word and sentence embeddings
- Transformer attention patterns
- Multilingual support features

### **Computer Vision** (`computer_vision.kpack`)
- Image feature representations
- Object detection knowledge
- Spatial attention mechanisms
- CNN feature maps

### **Mathematics** (`mathematics.kpack`)
- Mathematical concept embeddings
- Algebraic and calculus operations
- Probability and statistics knowledge
- Theorem and proof patterns

## Advanced Usage

### Custom Knowledge Sketches

```python
from core_nn.memory.kpack import KnowledgeSketch, SketchType, CompressionType

# Create custom attention sketch
attention_sketch = KnowledgeSketch(
    sketch_type=SketchType.ATTENTION,
    data=attention_weights,  # Your attention data
    compression=CompressionType.ZSTD,
    metadata={
        "layer": "transformer_layer_12",
        "heads": 16,
        "sequence_length": 512
    }
)

capsule.add_sketch(attention_sketch)
```

### Merging Capsules

```python
from core_nn.memory.kpack import merge_capsules

# Merge multiple capsules
merged = merge_capsules(
    [capsule1, capsule2, capsule3],
    name="Combined Knowledge",
    topic="multi_domain",
    description="Merged knowledge from multiple domains"
)
```

### Validation and Error Handling

```python
from core_nn.memory.kpack import validate_kpack

# Validate capsule file
is_valid, errors = validate_kpack("suspicious.kpack")
if not is_valid:
    print(f"Validation errors: {errors}")

# Validate capsule object
capsule_errors = capsule.validate()
if capsule_errors:
    print(f"Capsule errors: {capsule_errors}")
```

## Performance Characteristics

| Metric | Typical Range | Notes |
|--------|---------------|-------|
| **Compression Ratio** | 0.005 - 0.1 | 90-99.5% size reduction |
| **Load Time** | 10-100ms | Depends on capsule size |
| **Memory Usage** | 10-500MB | Uncompressed in memory |
| **Validation Time** | 1-10ms | Header and checksum only |

## Best Practices

### **Capsule Design**
- Use descriptive names and topics
- Include comprehensive metadata
- Add relevant tags for searchability
- Document provenance and source data

### **Storage Management**
- Organize capsules by topic/domain
- Use version numbers for updates
- Implement backup strategies
- Monitor disk space usage

### **MLCS Integration**
- Set appropriate memory limits
- Monitor compression statistics
- Use batch loading for efficiency
- Implement graceful error handling

### **Security**
- Validate all loaded capsules
- Use checksums for integrity
- Implement access controls
- Audit capsule sources

## Troubleshooting

### Common Issues

**"Invalid magic number"**
- File is corrupted or not a .kpack file
- Try re-downloading or recreating the capsule

**"Checksum verification failed"**
- File corruption during transfer
- Verify file integrity and source

**"Embedding dimensions mismatch"**
- MLCS input_dim doesn't match capsule embeddings
- Use dimension adaptation in MLCS integration

**"Memory limit exceeded"**
- Capsule too large for available memory
- Increase memory limits or use smaller capsules

### Debug Tools

```python
# Check capsule contents
capsule = load_kpack("debug.kpack")
print(f"Embeddings: {list(capsule.embeddings.keys())}")
print(f"Sketches: {len(capsule.sketches)}")
print(f"Size: {capsule.calculate_size()}")

# Validate format
is_valid, errors = validate_kpack("debug.kpack")
print(f"Valid: {is_valid}, Errors: {errors}")
```

## API Reference

### Core Functions
- `create_capsule()` - Create new knowledge capsule
- `save_kpack()` - Save capsule to file
- `load_kpack()` - Load capsule from file
- `validate_kpack()` - Validate capsule file
- `merge_capsules()` - Merge multiple capsules

### MLCS Integration
- `mlcs.load_knowledge_capsule()` - Load capsule into MLCS
- `mlcs.save_knowledge_capsule()` - Save MLCS pack as capsule
- `mlcs.export_learned_knowledge()` - Export all learned knowledge

### Data Classes
- `KnowledgeCapsule` - Main capsule container
- `CapsuleMetadata` - Metadata information
- `KnowledgeSketch` - Knowledge sketch representation

## Future Enhancements

- Encryption support for sensitive knowledge
- Network streaming for large capsules
- Advanced compression algorithms
- Content-based search and indexing
- Automatic knowledge extraction tools

---

For more information, see the [.kpack specification](kpack_specification.md) and [API documentation](../core_nn/memory/kpack.py).
