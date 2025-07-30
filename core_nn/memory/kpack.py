"""
Knowledge Capsule (.kpack) I/O Operations for CORE-NN

This module provides functionality for loading and saving knowledge capsules,
which are compressed representations of learned knowledge that can be
injected into or exported from the MLCS module.
"""

import json
import gzip
import zstandard as zstd
import struct
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import torch


class CompressionType(Enum):
    """Compression types for .kpack files."""
    NONE = 0
    GZIP = 1
    ZSTD = 2
    QUANTIZED = 3
    SPARSE = 4


class SketchType(Enum):
    """Types of knowledge sketches."""
    ATTENTION = "attention"
    GRADIENT = "gradient"
    ACTIVATION = "activation"
    EMBEDDING = "embedding"
    FEATURE_MAP = "feature_map"


@dataclass
class CapsuleMetadata:
    """Metadata for a knowledge capsule."""
    
    capsule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    topic: str = ""
    tags: List[str] = field(default_factory=list)
    creation_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modification_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    creator: str = "core-nn"
    version: str = "1.0.0"
    compression_level: float = 0.0
    original_size: int = 0
    compressed_size: int = 0
    embedding_dimensions: List[int] = field(default_factory=list)
    sketch_types: List[str] = field(default_factory=list)
    source_model: str = "core-nn-v1.0"
    
    # Compatibility information
    compatibility: Dict[str, Any] = field(default_factory=lambda: {
        "min_core_nn_version": "1.0.0",
        "required_features": ["mlcs", "rteu"]
    })
    
    # Statistical information
    statistics: Dict[str, Any] = field(default_factory=lambda: {
        "total_parameters": 0,
        "effective_rank": 0,
        "sparsity_ratio": 0.0,
        "knowledge_density": 0.0
    })
    
    # Provenance information
    provenance: Dict[str, Any] = field(default_factory=lambda: {
        "source_data": "",
        "training_steps": 0,
        "validation_score": 0.0,
        "extraction_method": "mlcs-compression"
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapsuleMetadata':
        """Create metadata from dictionary."""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate metadata and return list of errors."""
        errors = []
        
        if not self.capsule_id:
            errors.append("capsule_id is required")
        
        if not self.name:
            errors.append("name is required")
        
        if self.compression_level < 0 or self.compression_level > 1:
            errors.append("compression_level must be between 0 and 1")
        
        if self.original_size < 0:
            errors.append("original_size must be non-negative")
        
        if self.compressed_size < 0:
            errors.append("compressed_size must be non-negative")
        
        return errors


@dataclass
class KnowledgeSketch:
    """A knowledge sketch containing compressed representations."""
    
    sketch_type: SketchType
    data: Union[np.ndarray, torch.Tensor]
    compression: CompressionType = CompressionType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.sketch_type, str):
            self.sketch_type = SketchType(self.sketch_type)
        
        if isinstance(self.compression, int):
            self.compression = CompressionType(self.compression)
    
    def to_bytes(self) -> bytes:
        """Serialize sketch to bytes."""
        # Convert data to numpy array if it's a torch tensor
        if isinstance(self.data, torch.Tensor):
            data_array = self.data.detach().cpu().numpy()
        else:
            data_array = self.data
        
        # Create header
        header = struct.pack(
            '16sII8s',
            self.sketch_type.value.encode('utf-8').ljust(16, b'\0'),
            data_array.nbytes,
            self.compression.value,
            b'\0' * 8  # Reserved
        )
        
        # Serialize data based on compression type
        if self.compression == CompressionType.GZIP:
            data_bytes = gzip.compress(data_array.tobytes())
        elif self.compression == CompressionType.ZSTD:
            compressor = zstd.ZstdCompressor()
            data_bytes = compressor.compress(data_array.tobytes())
        else:
            data_bytes = data_array.tobytes()
        
        return header + data_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'KnowledgeSketch':
        """Deserialize sketch from bytes."""
        # Parse header
        header_size = 32
        header = data[:header_size]
        sketch_type_bytes, data_size, compression_value, _ = struct.unpack('16sII8s', header)
        
        sketch_type = SketchType(sketch_type_bytes.rstrip(b'\0').decode('utf-8'))
        compression = CompressionType(compression_value)
        
        # Extract and decompress data
        compressed_data = data[header_size:]
        
        if compression == CompressionType.GZIP:
            decompressed_data = gzip.decompress(compressed_data)
        elif compression == CompressionType.ZSTD:
            decompressor = zstd.ZstdDecompressor()
            decompressed_data = decompressor.decompress(compressed_data)
        else:
            decompressed_data = compressed_data
        
        # Convert back to numpy array (shape will need to be restored separately)
        data_array = np.frombuffer(decompressed_data, dtype=np.float32)
        
        return cls(
            sketch_type=sketch_type,
            data=data_array,
            compression=compression
        )


@dataclass
class KnowledgeCapsule:
    """A knowledge capsule containing embeddings, sketches, and metadata."""
    
    metadata: CapsuleMetadata
    embeddings: Dict[str, Union[np.ndarray, torch.Tensor]] = field(default_factory=dict)
    sketches: List[KnowledgeSketch] = field(default_factory=list)
    compression_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Update metadata with current information
        self.metadata.modification_time = datetime.now(timezone.utc).isoformat()
        self.metadata.sketch_types = [sketch.sketch_type.value for sketch in self.sketches]
        
        # Calculate embedding dimensions
        self.metadata.embedding_dimensions = []
        for embedding in self.embeddings.values():
            if isinstance(embedding, torch.Tensor):
                self.metadata.embedding_dimensions.append(embedding.shape[-1])
            elif isinstance(embedding, np.ndarray):
                self.metadata.embedding_dimensions.append(embedding.shape[-1])
    
    def add_embedding(self, name: str, embedding: Union[np.ndarray, torch.Tensor]) -> None:
        """Add an embedding to the capsule."""
        self.embeddings[name] = embedding
        self.__post_init__()  # Update metadata
    
    def add_sketch(self, sketch: KnowledgeSketch) -> None:
        """Add a knowledge sketch to the capsule."""
        self.sketches.append(sketch)
        self.__post_init__()  # Update metadata
    
    def get_embedding(self, name: str) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """Get an embedding by name."""
        return self.embeddings.get(name)
    
    def get_sketches_by_type(self, sketch_type: SketchType) -> List[KnowledgeSketch]:
        """Get all sketches of a specific type."""
        return [sketch for sketch in self.sketches if sketch.sketch_type == sketch_type]
    
    def calculate_size(self) -> Tuple[int, int]:
        """Calculate original and compressed sizes."""
        original_size = 0
        compressed_size = 0
        
        # Calculate embedding sizes
        for embedding in self.embeddings.values():
            if isinstance(embedding, torch.Tensor):
                original_size += embedding.numel() * embedding.element_size()
            elif isinstance(embedding, np.ndarray):
                original_size += embedding.nbytes
        
        # Calculate sketch sizes
        for sketch in self.sketches:
            sketch_bytes = sketch.to_bytes()
            compressed_size += len(sketch_bytes)
            
            if isinstance(sketch.data, torch.Tensor):
                original_size += sketch.data.numel() * sketch.data.element_size()
            elif isinstance(sketch.data, np.ndarray):
                original_size += sketch.data.nbytes
        
        # Add metadata size
        metadata_json = json.dumps(self.metadata.to_dict())
        original_size += len(metadata_json.encode('utf-8'))
        compressed_size += len(gzip.compress(metadata_json.encode('utf-8')))
        
        return original_size, compressed_size
    
    def validate(self) -> List[str]:
        """Validate the capsule and return list of errors."""
        errors = []
        
        # Validate metadata
        errors.extend(self.metadata.validate())
        
        # Validate embeddings
        if not self.embeddings:
            errors.append("At least one embedding is required")
        
        for name, embedding in self.embeddings.items():
            if not isinstance(embedding, (np.ndarray, torch.Tensor)):
                errors.append(f"Embedding '{name}' must be numpy array or torch tensor")
        
        # Validate sketches
        for i, sketch in enumerate(self.sketches):
            if not isinstance(sketch.data, (np.ndarray, torch.Tensor)):
                errors.append(f"Sketch {i} data must be numpy array or torch tensor")
        
        return errors


# File format constants
KPACK_MAGIC = b"KPACK001"
KPACK_VERSION = 1
HEADER_SIZE = 32


def _calculate_checksum(data: bytes) -> bytes:
    """Calculate SHA-256 checksum of data."""
    return hashlib.sha256(data).digest()


def _create_header(metadata_size: int, embeddings_size: int,
                  sketches_size: int, compression_size: int, flags: int = 0) -> bytes:
    """Create .kpack file header."""
    return struct.pack(
        '8sIIIIII',
        KPACK_MAGIC,
        KPACK_VERSION,
        metadata_size,
        embeddings_size,
        sketches_size,
        compression_size,
        flags
    )


def _parse_header(header: bytes) -> Dict[str, int]:
    """Parse .kpack file header."""
    if len(header) != HEADER_SIZE:
        raise ValueError(f"Invalid header size: {len(header)}, expected {HEADER_SIZE}")

    magic, version, metadata_size, embeddings_size, sketches_size, compression_size, flags = \
        struct.unpack('8sIIIIII', header)

    if magic != KPACK_MAGIC:
        raise ValueError(f"Invalid magic number: {magic}, expected {KPACK_MAGIC}")

    if version != KPACK_VERSION:
        raise ValueError(f"Unsupported version: {version}, expected {KPACK_VERSION}")

    return {
        'metadata_size': metadata_size,
        'embeddings_size': embeddings_size,
        'sketches_size': sketches_size,
        'compression_size': compression_size,
        'flags': flags
    }


def _serialize_embeddings(embeddings: Dict[str, Union[np.ndarray, torch.Tensor]]) -> bytes:
    """Serialize embeddings to bytes."""
    data = b''

    # Write level count
    data += struct.pack('I', len(embeddings))

    for name, embedding in embeddings.items():
        # Convert to numpy if needed
        if isinstance(embedding, torch.Tensor):
            array = embedding.detach().cpu().numpy()
        else:
            array = embedding

        # Write level header
        name_bytes = name.encode('utf-8')[:15]  # Limit name to 15 chars
        name_bytes = name_bytes.ljust(16, b'\0')

        header = struct.pack(
            '16sIII',
            name_bytes,
            array.shape[-1] if array.ndim > 0 else 1,  # dimension
            array.size,  # count
            array.dtype.itemsize  # data type size
        )
        data += header

        # Write data (compressed with zstd)
        compressor = zstd.ZstdCompressor()
        compressed_data = compressor.compress(array.tobytes())
        data += struct.pack('I', len(compressed_data))  # Add size prefix
        data += compressed_data

    return data


def _deserialize_embeddings(data: bytes) -> Dict[str, np.ndarray]:
    """Deserialize embeddings from bytes."""
    embeddings = {}
    offset = 0

    # Read level count
    level_count = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4

    decompressor = zstd.ZstdDecompressor()

    for _ in range(level_count):
        # Read level header
        header = data[offset:offset+28]
        name_bytes, dimension, count, dtype_size = struct.unpack('16sIII', header)
        offset += 28

        name = name_bytes.rstrip(b'\0').decode('utf-8')

        # Read compressed data size
        compressed_size = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # Read and decompress data
        compressed_data = data[offset:offset+compressed_size]
        decompressed_data = decompressor.decompress(compressed_data)
        offset += compressed_size

        # Reconstruct array
        if dtype_size == 4:
            dtype = np.float32
        elif dtype_size == 8:
            dtype = np.float64
        else:
            dtype = np.float32  # Default

        array = np.frombuffer(decompressed_data, dtype=dtype)

        # Try to reshape if we have dimension info
        if array.size > dimension and array.size % dimension == 0:
            array = array.reshape(-1, dimension)

        embeddings[name] = array

    return embeddings


def _serialize_sketches(sketches: List[KnowledgeSketch]) -> bytes:
    """Serialize sketches to bytes."""
    data = b''

    # Write sketch count
    data += struct.pack('I', len(sketches))

    # Write each sketch
    for sketch in sketches:
        sketch_bytes = sketch.to_bytes()
        data += struct.pack('I', len(sketch_bytes))  # Add size prefix
        data += sketch_bytes

    return data


def _deserialize_sketches(data: bytes) -> List[KnowledgeSketch]:
    """Deserialize sketches from bytes."""
    sketches = []
    offset = 0

    # Read sketch count
    sketch_count = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4

    for _ in range(sketch_count):
        # Read sketch size
        sketch_size = struct.unpack('I', data[offset:offset+4])[0]
        offset += 4

        # Read sketch data
        sketch_data = data[offset:offset+sketch_size]
        sketch = KnowledgeSketch.from_bytes(sketch_data)
        sketches.append(sketch)

        offset += sketch_size

    return sketches


def save_kpack(capsule: KnowledgeCapsule, path: Union[str, Path]) -> None:
    """
    Save a knowledge capsule to a .kpack file.

    Args:
        capsule: The knowledge capsule to save
        path: Path to save the .kpack file

    Raises:
        ValueError: If capsule validation fails
        IOError: If file cannot be written
    """
    # Validate capsule
    errors = capsule.validate()
    if errors:
        raise ValueError(f"Capsule validation failed: {', '.join(errors)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Update size information
    original_size, compressed_size = capsule.calculate_size()
    capsule.metadata.original_size = original_size
    capsule.metadata.compressed_size = compressed_size
    capsule.metadata.compression_level = 1.0 - (compressed_size / max(original_size, 1))

    # Serialize sections
    metadata_json = json.dumps(capsule.metadata.to_dict(), indent=2)
    metadata_bytes = gzip.compress(metadata_json.encode('utf-8'))

    embeddings_bytes = _serialize_embeddings(capsule.embeddings)
    sketches_bytes = _serialize_sketches(capsule.sketches)

    compression_json = json.dumps(capsule.compression_data, indent=2)
    compression_bytes = gzip.compress(compression_json.encode('utf-8'))

    # Create header
    header = _create_header(
        len(metadata_bytes),
        len(embeddings_bytes),
        len(sketches_bytes),
        len(compression_bytes)
    )

    # Combine all data
    file_data = header + metadata_bytes + embeddings_bytes + sketches_bytes + compression_bytes

    # Calculate and append checksum
    checksum = _calculate_checksum(file_data)
    file_data += checksum

    # Write to file
    with open(path, 'wb') as f:
        f.write(file_data)


def load_kpack(path: Union[str, Path]) -> KnowledgeCapsule:
    """
    Load a knowledge capsule from a .kpack file.

    Args:
        path: Path to the .kpack file

    Returns:
        The loaded knowledge capsule

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'rb') as f:
        file_data = f.read()

    # Verify checksum
    if len(file_data) < HEADER_SIZE + 32:  # Header + checksum
        raise ValueError("File too small to be valid .kpack")

    data_without_checksum = file_data[:-32]
    stored_checksum = file_data[-32:]
    calculated_checksum = _calculate_checksum(data_without_checksum)

    if stored_checksum != calculated_checksum:
        raise ValueError("Checksum verification failed - file may be corrupted")

    # Parse header
    header = file_data[:HEADER_SIZE]
    header_info = _parse_header(header)

    offset = HEADER_SIZE

    # Read metadata
    metadata_size = header_info['metadata_size']
    metadata_bytes = file_data[offset:offset + metadata_size]
    metadata_json = gzip.decompress(metadata_bytes).decode('utf-8')
    metadata_dict = json.loads(metadata_json)
    metadata = CapsuleMetadata.from_dict(metadata_dict)
    offset += metadata_size

    # Read embeddings
    embeddings_size = header_info['embeddings_size']
    embeddings_bytes = file_data[offset:offset + embeddings_size]
    embeddings = _deserialize_embeddings(embeddings_bytes)
    offset += embeddings_size

    # Read sketches
    sketches_size = header_info['sketches_size']
    sketches_bytes = file_data[offset:offset + sketches_size]
    sketches = _deserialize_sketches(sketches_bytes)
    offset += sketches_size

    # Read compression data
    compression_size = header_info['compression_size']
    compression_bytes = file_data[offset:offset + compression_size]
    compression_json = gzip.decompress(compression_bytes).decode('utf-8')
    compression_data = json.loads(compression_json)

    # Create capsule
    capsule = KnowledgeCapsule(
        metadata=metadata,
        embeddings=embeddings,
        sketches=sketches,
        compression_data=compression_data
    )

    return capsule


def validate_kpack(path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate a .kpack file without fully loading it.

    Args:
        path: Path to the .kpack file

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        path = Path(path)
        if not path.exists():
            errors.append(f"File not found: {path}")
            return False, errors

        with open(path, 'rb') as f:
            # Check file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to start

            if file_size < HEADER_SIZE + 32:
                errors.append("File too small to be valid .kpack")
                return False, errors

            # Read and validate header
            header = f.read(HEADER_SIZE)
            try:
                header_info = _parse_header(header)
            except ValueError as e:
                errors.append(f"Invalid header: {e}")
                return False, errors

            # Check if file size matches header information
            expected_size = (HEADER_SIZE +
                           header_info['metadata_size'] +
                           header_info['embeddings_size'] +
                           header_info['sketches_size'] +
                           header_info['compression_size'] +
                           32)  # checksum

            if file_size != expected_size:
                errors.append(f"File size mismatch: expected {expected_size}, got {file_size}")
                return False, errors

            # Verify checksum
            f.seek(0)
            file_data = f.read()
            data_without_checksum = file_data[:-32]
            stored_checksum = file_data[-32:]
            calculated_checksum = _calculate_checksum(data_without_checksum)

            if stored_checksum != calculated_checksum:
                errors.append("Checksum verification failed")
                return False, errors

        return True, []

    except Exception as e:
        errors.append(f"Validation error: {e}")
        return False, errors


def create_capsule(name: str, topic: str, description: str = "",
                  embeddings: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
                  sketches: Optional[List[KnowledgeSketch]] = None,
                  **metadata_kwargs) -> KnowledgeCapsule:
    """
    Create a new knowledge capsule with the given parameters.

    Args:
        name: Human-readable name for the capsule
        topic: Primary topic category
        description: Detailed description
        embeddings: Dictionary of embeddings to include
        sketches: List of knowledge sketches to include
        **metadata_kwargs: Additional metadata fields

    Returns:
        A new KnowledgeCapsule instance
    """
    metadata = CapsuleMetadata(
        name=name,
        topic=topic,
        description=description,
        **metadata_kwargs
    )

    capsule = KnowledgeCapsule(
        metadata=metadata,
        embeddings=embeddings or {},
        sketches=sketches or []
    )

    return capsule


def merge_capsules(capsules: List[KnowledgeCapsule],
                  name: str, topic: str, description: str = "") -> KnowledgeCapsule:
    """
    Merge multiple knowledge capsules into a single capsule.

    Args:
        capsules: List of capsules to merge
        name: Name for the merged capsule
        topic: Topic for the merged capsule
        description: Description for the merged capsule

    Returns:
        A new merged KnowledgeCapsule

    Raises:
        ValueError: If no capsules provided or capsules are invalid
    """
    if not capsules:
        raise ValueError("At least one capsule must be provided")

    # Validate all capsules
    for i, capsule in enumerate(capsules):
        errors = capsule.validate()
        if errors:
            raise ValueError(f"Capsule {i} validation failed: {', '.join(errors)}")

    # Create merged metadata
    merged_metadata = CapsuleMetadata(
        name=name,
        topic=topic,
        description=description,
        tags=list(set(tag for capsule in capsules for tag in capsule.metadata.tags)),
        creator="core-nn-merger"
    )

    # Merge embeddings (with name prefixing to avoid conflicts)
    merged_embeddings = {}
    for i, capsule in enumerate(capsules):
        for emb_name, embedding in capsule.embeddings.items():
            merged_name = f"capsule_{i}_{emb_name}"
            merged_embeddings[merged_name] = embedding

    # Merge sketches
    merged_sketches = []
    for capsule in capsules:
        merged_sketches.extend(capsule.sketches)

    # Merge compression data
    merged_compression_data = {
        "source_capsules": [capsule.metadata.capsule_id for capsule in capsules],
        "merge_timestamp": datetime.now(timezone.utc).isoformat(),
        "merge_method": "simple_concatenation"
    }

    merged_capsule = KnowledgeCapsule(
        metadata=merged_metadata,
        embeddings=merged_embeddings,
        sketches=merged_sketches,
        compression_data=merged_compression_data
    )

    return merged_capsule


# Convenience functions for MLCS integration
def load_kpack_for_mlcs(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a .kpack file and return data in MLCS-compatible format.

    Args:
        path: Path to the .kpack file

    Returns:
        Dictionary with MLCS-compatible data structure
    """
    capsule = load_kpack(path)

    return {
        'embeddings': capsule.embeddings,
        'sketches': {sketch.sketch_type.value: sketch.data for sketch in capsule.sketches},
        'metadata': capsule.metadata.to_dict(),
        'compression_data': capsule.compression_data
    }


def save_kpack_from_mlcs(mlcs_data: Dict[str, Any], path: Union[str, Path],
                        name: str, topic: str, description: str = "") -> None:
    """
    Save MLCS data as a .kpack file.

    Args:
        mlcs_data: Data from MLCS in expected format
        path: Path to save the .kpack file
        name: Name for the capsule
        topic: Topic for the capsule
        description: Description for the capsule
    """
    # Create sketches from MLCS data
    sketches = []
    if 'sketches' in mlcs_data:
        for sketch_type_str, data in mlcs_data['sketches'].items():
            try:
                sketch_type = SketchType(sketch_type_str)
                sketch = KnowledgeSketch(
                    sketch_type=sketch_type,
                    data=data,
                    compression=CompressionType.ZSTD
                )
                sketches.append(sketch)
            except ValueError:
                # Skip unknown sketch types
                continue

    # Create capsule
    capsule = create_capsule(
        name=name,
        topic=topic,
        description=description,
        embeddings=mlcs_data.get('embeddings', {}),
        sketches=sketches
    )

    # Add MLCS-specific compression data
    if 'compression_data' in mlcs_data:
        capsule.compression_data.update(mlcs_data['compression_data'])

    # Save capsule
    save_kpack(capsule, path)
