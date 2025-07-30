# .kpack Knowledge Capsule File Format Specification

## Overview

The `.kpack` format is a compressed, self-contained knowledge capsule format designed for CORE-NN's Multi-Level Compression Synthesizer (MLCS). It enables efficient storage, transfer, and loading of learned knowledge representations.

## File Structure

```
.kpack File Layout:
┌─────────────────────────────────────┐
│ Header (32 bytes)                   │
├─────────────────────────────────────┤
│ Metadata Section (variable)         │
├─────────────────────────────────────┤
│ Embeddings Section (variable)       │
├─────────────────────────────────────┤
│ Sketches Section (variable)         │
├─────────────────────────────────────┤
│ Compression Data (variable)         │
├─────────────────────────────────────┤
│ Checksum (32 bytes)                 │
└─────────────────────────────────────┘
```

## Header Format (32 bytes)

```
Offset | Size | Field           | Description
-------|------|-----------------|----------------------------------
0      | 8    | Magic           | "KPACK001" (version identifier)
8      | 4    | Format Version  | uint32 format version number
12     | 4    | Metadata Size   | uint32 metadata section size
16     | 4    | Embeddings Size | uint32 embeddings section size  
20     | 4    | Sketches Size   | uint32 sketches section size
24     | 4    | Compression Size| uint32 compression data size
28     | 4    | Flags           | uint32 feature flags
```

## Metadata Schema

```json
{
  "capsule_id": "uuid-string",
  "name": "human-readable-name",
  "description": "detailed-description",
  "topic": "primary-topic-category",
  "tags": ["tag1", "tag2", "tag3"],
  "creation_time": "iso-8601-timestamp",
  "modification_time": "iso-8601-timestamp",
  "creator": "creator-identifier",
  "version": "semantic-version",
  "compression_level": 0.75,
  "original_size": 1048576,
  "compressed_size": 262144,
  "embedding_dimensions": [768, 512, 256],
  "sketch_types": ["attention", "gradient", "activation"],
  "source_model": "core-nn-v1.0",
  "compatibility": {
    "min_core_nn_version": "1.0.0",
    "required_features": ["mlcs", "rteu"]
  },
  "statistics": {
    "total_parameters": 1000000,
    "effective_rank": 128,
    "sparsity_ratio": 0.85,
    "knowledge_density": 0.92
  },
  "provenance": {
    "source_data": "training-corpus-v2",
    "training_steps": 50000,
    "validation_score": 0.94,
    "extraction_method": "mlcs-compression"
  }
}
```

## Embeddings Section

Stores multi-level embedding representations:

```
Embeddings Format:
┌─────────────────────────────────────┐
│ Level Count (4 bytes)               │
├─────────────────────────────────────┤
│ Level 0 Header (16 bytes)           │
│ - Dimension (4 bytes)               │
│ - Count (4 bytes)                   │
│ - Data Type (4 bytes)               │
│ - Compression (4 bytes)             │
├─────────────────────────────────────┤
│ Level 0 Data (variable)             │
├─────────────────────────────────────┤
│ Level 1 Header (16 bytes)           │
├─────────────────────────────────────┤
│ Level 1 Data (variable)             │
├─────────────────────────────────────┤
│ ... (additional levels)             │
└─────────────────────────────────────┘
```

## Sketches Section

Stores compressed knowledge sketches:

```
Sketches Format:
┌─────────────────────────────────────┐
│ Sketch Count (4 bytes)              │
├─────────────────────────────────────┤
│ Sketch 0 Header (32 bytes)          │
│ - Type (16 bytes, null-terminated)  │
│ - Size (4 bytes)                    │
│ - Compression (4 bytes)             │
│ - Reserved (8 bytes)                │
├─────────────────────────────────────┤
│ Sketch 0 Data (variable)            │
├─────────────────────────────────────┤
│ ... (additional sketches)           │
└─────────────────────────────────────┘
```

## Compression Data Section

Stores MLCS-specific compression artifacts:

```json
{
  "quantization_levels": [
    {
      "level": 0,
      "codebook_size": 256,
      "embedding_dim": 768,
      "quantization_error": 0.02
    }
  ],
  "compression_tree": {
    "root": "node-id",
    "nodes": {
      "node-id": {
        "type": "compression",
        "children": ["child-1", "child-2"],
        "data": "compressed-representation"
      }
    }
  },
  "reconstruction_hints": {
    "priority_indices": [1, 5, 12, 23],
    "attention_masks": "base64-encoded-data",
    "layer_importance": [0.9, 0.8, 0.7, 0.6]
  }
}
```

## Feature Flags

```
Bit | Feature                    | Description
----|----------------------------|----------------------------------
0   | COMPRESSED_EMBEDDINGS      | Embeddings use compression
1   | ENCRYPTED_DATA             | Data is encrypted
2   | DIFFERENTIAL_COMPRESSION   | Uses differential compression
3   | SPARSE_REPRESENTATION      | Uses sparse tensor format
4   | QUANTIZED_WEIGHTS          | Weights are quantized
5   | HIERARCHICAL_STRUCTURE     | Multi-level hierarchy
6   | ATTENTION_SKETCHES         | Contains attention sketches
7   | GRADIENT_SKETCHES          | Contains gradient sketches
8-31| RESERVED                   | Reserved for future use
```

## Compression Methods

1. **GZIP**: Standard compression for metadata and small data
2. **ZSTD**: High-performance compression for large tensors
3. **Quantization**: 8-bit/16-bit quantization for embeddings
4. **Sparse**: CSR/COO format for sparse tensors
5. **Delta**: Differential compression for similar embeddings

## Validation and Integrity

- **CRC32 Checksum**: 32-byte checksum of entire file content
- **Magic Number**: Validates file format
- **Version Check**: Ensures compatibility
- **Size Validation**: Verifies section sizes match header
- **Schema Validation**: JSON schema validation for metadata

## Usage Examples

### Loading a Knowledge Capsule
```python
from core_nn.memory.kpack import load_kpack

# Load capsule
capsule = load_kpack("knowledge/ml_fundamentals.kpack")

# Access data
embeddings = capsule.embeddings
sketches = capsule.sketches
metadata = capsule.metadata
```

### Saving a Knowledge Capsule
```python
from core_nn.memory.kpack import save_kpack

# Create and save capsule
save_kpack(
    topic="machine_learning",
    embeddings=learned_embeddings,
    sketches=attention_sketches,
    metadata={"description": "ML fundamentals"},
    path="output/ml_capsule.kpack"
)
```

## Compatibility

- **Minimum CORE-NN Version**: 1.0.0
- **Python Version**: 3.8+
- **Dependencies**: numpy, torch, zstandard
- **File Size Limits**: 2GB per capsule (practical limit)
- **Platform**: Cross-platform (Windows, Linux, macOS)

## Security Considerations

- **Validation**: All inputs validated before processing
- **Sandboxing**: Deserialization in controlled environment
- **Size Limits**: Prevents memory exhaustion attacks
- **Type Safety**: Strict type checking during load
- **Checksum Verification**: Detects corruption/tampering
