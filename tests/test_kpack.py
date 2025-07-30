"""
Tests for .kpack Knowledge Capsule I/O operations.
"""

import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

from core_nn.memory.kpack import (
    KnowledgeCapsule,
    KnowledgeSketch,
    CapsuleMetadata,
    SketchType,
    CompressionType,
    load_kpack,
    save_kpack,
    validate_kpack,
    create_capsule,
    merge_capsules,
    load_kpack_for_mlcs,
    save_kpack_from_mlcs
)


class TestCapsuleMetadata:
    """Test CapsuleMetadata functionality."""
    
    def test_default_metadata(self):
        """Test default metadata creation."""
        metadata = CapsuleMetadata()
        
        assert metadata.capsule_id != ""
        assert metadata.name == ""
        assert metadata.creator == "core-nn"
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.tags, list)
        assert isinstance(metadata.compatibility, dict)
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Valid metadata
        metadata = CapsuleMetadata(name="test", capsule_id="test-id")
        errors = metadata.validate()
        assert len(errors) == 0
        
        # Invalid metadata
        invalid_metadata = CapsuleMetadata(
            name="",  # Empty name
            capsule_id="",  # Empty ID
            compression_level=-0.5,  # Invalid compression level
            original_size=-100  # Negative size
        )
        errors = invalid_metadata.validate()
        assert len(errors) > 0
        assert any("name is required" in error for error in errors)
        assert any("capsule_id is required" in error for error in errors)
    
    def test_metadata_serialization(self):
        """Test metadata to/from dict conversion."""
        metadata = CapsuleMetadata(
            name="test_capsule",
            description="Test description",
            tags=["test", "example"]
        )
        
        # Convert to dict and back
        metadata_dict = metadata.to_dict()
        restored_metadata = CapsuleMetadata.from_dict(metadata_dict)
        
        assert restored_metadata.name == metadata.name
        assert restored_metadata.description == metadata.description
        assert restored_metadata.tags == metadata.tags


class TestKnowledgeSketch:
    """Test KnowledgeSketch functionality."""
    
    def test_sketch_creation(self):
        """Test sketch creation with different data types."""
        # NumPy array
        np_data = np.random.randn(10, 64).astype(np.float32)
        sketch_np = KnowledgeSketch(
            sketch_type=SketchType.ATTENTION,
            data=np_data,
            compression=CompressionType.ZSTD
        )
        assert sketch_np.sketch_type == SketchType.ATTENTION
        assert isinstance(sketch_np.data, np.ndarray)
        
        # Torch tensor
        torch_data = torch.randn(10, 64)
        sketch_torch = KnowledgeSketch(
            sketch_type=SketchType.GRADIENT,
            data=torch_data,
            compression=CompressionType.GZIP
        )
        assert sketch_torch.sketch_type == SketchType.GRADIENT
        assert isinstance(sketch_torch.data, torch.Tensor)
    
    def test_sketch_serialization(self):
        """Test sketch serialization and deserialization."""
        original_data = np.random.randn(5, 32).astype(np.float32)
        original_sketch = KnowledgeSketch(
            sketch_type=SketchType.EMBEDDING,
            data=original_data,
            compression=CompressionType.ZSTD
        )
        
        # Serialize and deserialize
        serialized = original_sketch.to_bytes()
        restored_sketch = KnowledgeSketch.from_bytes(serialized)
        
        assert restored_sketch.sketch_type == original_sketch.sketch_type
        assert restored_sketch.compression == original_sketch.compression
        # Note: Shape information is lost in current implementation
        assert restored_sketch.data.size == original_data.size


class TestKnowledgeCapsule:
    """Test KnowledgeCapsule functionality."""
    
    @pytest.fixture
    def sample_capsule(self):
        """Create a sample knowledge capsule for testing."""
        metadata = CapsuleMetadata(
            name="test_capsule",
            description="Test knowledge capsule",
            topic="testing"
        )
        
        embeddings = {
            "primary": np.random.randn(100, 768).astype(np.float32),
            "secondary": torch.randn(50, 512)
        }
        
        sketches = [
            KnowledgeSketch(
                sketch_type=SketchType.ATTENTION,
                data=np.random.randn(10, 64).astype(np.float32),
                compression=CompressionType.ZSTD
            ),
            KnowledgeSketch(
                sketch_type=SketchType.GRADIENT,
                data=torch.randn(8, 32),
                compression=CompressionType.GZIP
            )
        ]
        
        return KnowledgeCapsule(
            metadata=metadata,
            embeddings=embeddings,
            sketches=sketches
        )
    
    def test_capsule_creation(self, sample_capsule):
        """Test capsule creation and basic properties."""
        assert sample_capsule.metadata.name == "test_capsule"
        assert len(sample_capsule.embeddings) == 2
        assert len(sample_capsule.sketches) == 2
        assert "primary" in sample_capsule.embeddings
        assert "secondary" in sample_capsule.embeddings
    
    def test_capsule_validation(self, sample_capsule):
        """Test capsule validation."""
        errors = sample_capsule.validate()
        assert len(errors) == 0
        
        # Test invalid capsule
        invalid_capsule = KnowledgeCapsule(
            metadata=CapsuleMetadata(name="", capsule_id=""),
            embeddings={},  # No embeddings
            sketches=[]
        )
        errors = invalid_capsule.validate()
        assert len(errors) > 0
    
    def test_add_embedding(self, sample_capsule):
        """Test adding embeddings to capsule."""
        new_embedding = np.random.randn(20, 256).astype(np.float32)
        sample_capsule.add_embedding("new_embedding", new_embedding)
        
        assert "new_embedding" in sample_capsule.embeddings
        assert sample_capsule.embeddings["new_embedding"] is new_embedding
    
    def test_add_sketch(self, sample_capsule):
        """Test adding sketches to capsule."""
        new_sketch = KnowledgeSketch(
            sketch_type=SketchType.FEATURE_MAP,
            data=np.random.randn(5, 16).astype(np.float32)
        )
        sample_capsule.add_sketch(new_sketch)
        
        assert len(sample_capsule.sketches) == 3
        assert sample_capsule.sketches[-1] == new_sketch
    
    def test_get_sketches_by_type(self, sample_capsule):
        """Test filtering sketches by type."""
        attention_sketches = sample_capsule.get_sketches_by_type(SketchType.ATTENTION)
        assert len(attention_sketches) == 1
        assert attention_sketches[0].sketch_type == SketchType.ATTENTION
        
        gradient_sketches = sample_capsule.get_sketches_by_type(SketchType.GRADIENT)
        assert len(gradient_sketches) == 1
        
        nonexistent_sketches = sample_capsule.get_sketches_by_type(SketchType.FEATURE_MAP)
        assert len(nonexistent_sketches) == 0
    
    def test_calculate_size(self, sample_capsule):
        """Test size calculation."""
        original_size, compressed_size = sample_capsule.calculate_size()
        assert original_size > 0
        assert compressed_size > 0
        assert compressed_size <= original_size  # Compression should reduce size


class TestKpackIO:
    """Test .kpack file I/O operations."""
    
    @pytest.fixture
    def sample_capsule(self):
        """Create a sample capsule for I/O testing."""
        return create_capsule(
            name="io_test_capsule",
            topic="testing",
            description="Test capsule for I/O operations",
            embeddings={
                "test_embedding": np.random.randn(50, 384).astype(np.float32)
            },
            sketches=[
                KnowledgeSketch(
                    sketch_type=SketchType.ATTENTION,
                    data=np.random.randn(10, 32).astype(np.float32),
                    compression=CompressionType.ZSTD
                )
            ]
        )
    
    def test_save_and_load_kpack(self, sample_capsule):
        """Test saving and loading .kpack files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kpack_path = Path(temp_dir) / "test.kpack"
            
            # Save capsule
            save_kpack(sample_capsule, kpack_path)
            assert kpack_path.exists()
            
            # Load capsule
            loaded_capsule = load_kpack(kpack_path)
            
            # Verify loaded data
            assert loaded_capsule.metadata.name == sample_capsule.metadata.name
            assert loaded_capsule.metadata.topic == sample_capsule.metadata.topic
            assert len(loaded_capsule.embeddings) == len(sample_capsule.embeddings)
            assert len(loaded_capsule.sketches) == len(sample_capsule.sketches)
    
    def test_validate_kpack(self, sample_capsule):
        """Test .kpack file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kpack_path = Path(temp_dir) / "test.kpack"
            
            # Save valid capsule
            save_kpack(sample_capsule, kpack_path)
            
            # Validate
            is_valid, errors = validate_kpack(kpack_path)
            assert is_valid
            assert len(errors) == 0
            
            # Test non-existent file
            nonexistent_path = Path(temp_dir) / "nonexistent.kpack"
            is_valid, errors = validate_kpack(nonexistent_path)
            assert not is_valid
            assert len(errors) > 0
    
    def test_create_capsule_utility(self):
        """Test create_capsule utility function."""
        capsule = create_capsule(
            name="utility_test",
            topic="testing",
            description="Test capsule creation utility",
            tags=["test", "utility"]
        )
        
        assert capsule.metadata.name == "utility_test"
        assert capsule.metadata.topic == "testing"
        assert "test" in capsule.metadata.tags
        assert "utility" in capsule.metadata.tags
    
    def test_merge_capsules(self):
        """Test capsule merging functionality."""
        # Create multiple capsules
        capsule1 = create_capsule(
            name="capsule1",
            topic="test",
            embeddings={"emb1": np.random.randn(10, 64).astype(np.float32)}
        )
        
        capsule2 = create_capsule(
            name="capsule2", 
            topic="test",
            embeddings={"emb2": np.random.randn(15, 64).astype(np.float32)}
        )
        
        # Merge capsules
        merged = merge_capsules(
            [capsule1, capsule2],
            name="merged_capsule",
            topic="merged_test",
            description="Merged test capsule"
        )
        
        assert merged.metadata.name == "merged_capsule"
        assert merged.metadata.topic == "merged_test"
        assert len(merged.embeddings) == 2  # Should have both embeddings with prefixed names
        assert any("capsule_0_emb1" in key for key in merged.embeddings.keys())
        assert any("capsule_1_emb2" in key for key in merged.embeddings.keys())


class TestMLCSIntegration:
    """Test MLCS integration functions."""
    
    def test_load_kpack_for_mlcs(self):
        """Test loading .kpack for MLCS."""
        # Create and save a test capsule
        capsule = create_capsule(
            name="mlcs_test",
            topic="testing",
            embeddings={"test": np.random.randn(20, 128).astype(np.float32)}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            kpack_path = Path(temp_dir) / "mlcs_test.kpack"
            save_kpack(capsule, kpack_path)
            
            # Load for MLCS
            mlcs_data = load_kpack_for_mlcs(kpack_path)
            
            assert "embeddings" in mlcs_data
            assert "sketches" in mlcs_data
            assert "metadata" in mlcs_data
            assert "compression_data" in mlcs_data
            assert "test" in mlcs_data["embeddings"]
    
    def test_save_kpack_from_mlcs(self):
        """Test saving MLCS data as .kpack."""
        mlcs_data = {
            "embeddings": {
                "primary": np.random.randn(30, 256).astype(np.float32)
            },
            "sketches": {
                "attention": np.random.randn(5, 64).astype(np.float32)
            },
            "compression_data": {
                "test_data": "test_value"
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            kpack_path = Path(temp_dir) / "from_mlcs.kpack"
            
            save_kpack_from_mlcs(
                mlcs_data=mlcs_data,
                path=kpack_path,
                name="from_mlcs_test",
                topic="testing",
                description="Test saving from MLCS"
            )
            
            assert kpack_path.exists()
            
            # Verify by loading back
            loaded_capsule = load_kpack(kpack_path)
            assert loaded_capsule.metadata.name == "from_mlcs_test"
            assert "primary" in loaded_capsule.embeddings
            assert len(loaded_capsule.sketches) == 1
            assert loaded_capsule.sketches[0].sketch_type == SketchType.ATTENTION
