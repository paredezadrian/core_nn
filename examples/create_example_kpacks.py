#!/usr/bin/env python3
"""
Create example .kpack knowledge capsules for demonstration.

This script creates several example knowledge capsules that demonstrate
different use cases and features of the .kpack format.
"""

import numpy as np
import torch
from pathlib import Path

from core_nn.memory.kpack import (
    create_capsule,
    save_kpack,
    KnowledgeSketch,
    SketchType,
    CompressionType
)


def create_ml_fundamentals_capsule():
    """Create a knowledge capsule for machine learning fundamentals."""
    
    # Simulate embeddings for ML concepts
    concept_embeddings = {
        "supervised_learning": np.random.randn(100, 768).astype(np.float32),
        "unsupervised_learning": np.random.randn(80, 768).astype(np.float32),
        "neural_networks": np.random.randn(150, 768).astype(np.float32),
        "gradient_descent": np.random.randn(60, 768).astype(np.float32),
        "backpropagation": np.random.randn(90, 768).astype(np.float32)
    }
    
    # Create attention sketches for important concepts
    attention_sketches = [
        KnowledgeSketch(
            sketch_type=SketchType.ATTENTION,
            data=np.random.randn(20, 64).astype(np.float32),
            compression=CompressionType.ZSTD,
            metadata={"concept": "neural_network_attention", "importance": 0.9}
        ),
        KnowledgeSketch(
            sketch_type=SketchType.GRADIENT,
            data=np.random.randn(15, 32).astype(np.float32),
            compression=CompressionType.ZSTD,
            metadata={"concept": "optimization_gradients", "importance": 0.8}
        )
    ]
    
    capsule = create_capsule(
        name="ML Fundamentals",
        topic="machine_learning",
        description="Core concepts and knowledge for machine learning fundamentals including supervised/unsupervised learning, neural networks, and optimization techniques.",
        embeddings=concept_embeddings,
        sketches=attention_sketches,
        tags=["machine_learning", "fundamentals", "education", "concepts"],
        creator="core-nn-examples",
        provenance={
            "source_data": "ML textbooks and papers",
            "training_steps": 10000,
            "validation_score": 0.92,
            "extraction_method": "concept_distillation"
        }
    )
    
    return capsule


def create_nlp_knowledge_capsule():
    """Create a knowledge capsule for natural language processing."""
    
    # NLP-specific embeddings
    nlp_embeddings = {
        "word_embeddings": np.random.randn(200, 512).astype(np.float32),
        "sentence_embeddings": np.random.randn(100, 768).astype(np.float32),
        "transformer_representations": torch.randn(150, 1024),
        "attention_patterns": np.random.randn(80, 256).astype(np.float32)
    }
    
    # NLP-specific sketches
    nlp_sketches = [
        KnowledgeSketch(
            sketch_type=SketchType.ATTENTION,
            data=np.random.randn(50, 128).astype(np.float32),
            compression=CompressionType.ZSTD,
            metadata={"layer": "transformer_attention", "heads": 12}
        ),
        KnowledgeSketch(
            sketch_type=SketchType.EMBEDDING,
            data=np.random.randn(300, 64).astype(np.float32),
            compression=CompressionType.ZSTD,
            metadata={"type": "contextual_embeddings", "vocab_size": 50000}
        ),
        KnowledgeSketch(
            sketch_type=SketchType.FEATURE_MAP,
            data=np.random.randn(25, 256).astype(np.float32),
            compression=CompressionType.GZIP,
            metadata={"feature_type": "linguistic_features", "languages": ["en", "es", "fr"]}
        )
    ]
    
    capsule = create_capsule(
        name="NLP Knowledge Base",
        topic="natural_language_processing",
        description="Comprehensive knowledge base for natural language processing including word embeddings, transformer models, attention mechanisms, and multilingual representations.",
        embeddings=nlp_embeddings,
        sketches=nlp_sketches,
        tags=["nlp", "transformers", "embeddings", "attention", "multilingual"],
        creator="core-nn-nlp-team",
        statistics={
            "total_parameters": 2500000,
            "effective_rank": 256,
            "sparsity_ratio": 0.75,
            "knowledge_density": 0.88
        },
        provenance={
            "source_data": "Common Crawl + Wikipedia + Books",
            "training_steps": 100000,
            "validation_score": 0.94,
            "extraction_method": "transformer_distillation"
        }
    )
    
    return capsule


def create_computer_vision_capsule():
    """Create a knowledge capsule for computer vision."""
    
    # Computer vision embeddings
    cv_embeddings = {
        "image_features": np.random.randn(500, 2048).astype(np.float32),
        "object_representations": np.random.randn(200, 1024).astype(np.float32),
        "scene_embeddings": torch.randn(150, 512),
        "texture_features": np.random.randn(100, 256).astype(np.float32)
    }
    
    # Computer vision sketches
    cv_sketches = [
        KnowledgeSketch(
            sketch_type=SketchType.FEATURE_MAP,
            data=np.random.randn(64, 64, 32).astype(np.float32).reshape(-1, 32),
            compression=CompressionType.ZSTD,
            metadata={"layer": "conv_features", "resolution": "64x64", "channels": 32}
        ),
        KnowledgeSketch(
            sketch_type=SketchType.ATTENTION,
            data=np.random.randn(100, 196).astype(np.float32),  # 14x14 spatial attention
            compression=CompressionType.ZSTD,
            metadata={"type": "spatial_attention", "resolution": "14x14"}
        ),
        KnowledgeSketch(
            sketch_type=SketchType.ACTIVATION,
            data=np.random.randn(50, 512).astype(np.float32),
            compression=CompressionType.GZIP,
            metadata={"layer": "final_features", "activation": "relu"}
        )
    ]
    
    capsule = create_capsule(
        name="Computer Vision Knowledge",
        topic="computer_vision",
        description="Comprehensive computer vision knowledge including image features, object detection, scene understanding, and visual attention mechanisms.",
        embeddings=cv_embeddings,
        sketches=cv_sketches,
        tags=["computer_vision", "cnn", "object_detection", "image_classification", "features"],
        creator="core-nn-vision-team",
        statistics={
            "total_parameters": 5000000,
            "effective_rank": 512,
            "sparsity_ratio": 0.60,
            "knowledge_density": 0.91
        },
        provenance={
            "source_data": "ImageNet + COCO + OpenImages",
            "training_steps": 200000,
            "validation_score": 0.96,
            "extraction_method": "cnn_feature_distillation"
        }
    )
    
    return capsule


def create_mathematics_capsule():
    """Create a knowledge capsule for mathematical concepts."""
    
    # Mathematical concept embeddings
    math_embeddings = {
        "algebra_concepts": np.random.randn(80, 384).astype(np.float32),
        "calculus_operations": np.random.randn(120, 384).astype(np.float32),
        "linear_algebra": np.random.randn(100, 384).astype(np.float32),
        "probability_theory": np.random.randn(90, 384).astype(np.float32),
        "statistics_methods": torch.randn(70, 384)
    }
    
    # Mathematical reasoning sketches
    math_sketches = [
        KnowledgeSketch(
            sketch_type=SketchType.EMBEDDING,
            data=np.random.randn(50, 128).astype(np.float32),
            compression=CompressionType.ZSTD,
            metadata={"type": "theorem_embeddings", "domain": "analysis"}
        ),
        KnowledgeSketch(
            sketch_type=SketchType.GRADIENT,
            data=np.random.randn(30, 64).astype(np.float32),
            compression=CompressionType.GZIP,
            metadata={"type": "proof_gradients", "reasoning_depth": 5}
        )
    ]
    
    capsule = create_capsule(
        name="Mathematical Knowledge",
        topic="mathematics",
        description="Core mathematical knowledge including algebra, calculus, linear algebra, probability, and statistics with embedded reasoning patterns.",
        embeddings=math_embeddings,
        sketches=math_sketches,
        tags=["mathematics", "algebra", "calculus", "probability", "reasoning"],
        creator="core-nn-math-team",
        statistics={
            "total_parameters": 800000,
            "effective_rank": 128,
            "sparsity_ratio": 0.85,
            "knowledge_density": 0.95
        },
        provenance={
            "source_data": "Mathematical textbooks + arXiv papers",
            "training_steps": 50000,
            "validation_score": 0.89,
            "extraction_method": "symbolic_reasoning_distillation"
        }
    )
    
    return capsule


def main():
    """Create and save all example knowledge capsules."""
    
    # Create output directory
    output_dir = Path("examples/kpacks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating example knowledge capsules...")
    
    # Create capsules
    capsules = [
        ("ml_fundamentals.kpack", create_ml_fundamentals_capsule()),
        ("nlp_knowledge.kpack", create_nlp_knowledge_capsule()),
        ("computer_vision.kpack", create_computer_vision_capsule()),
        ("mathematics.kpack", create_mathematics_capsule())
    ]
    
    # Save capsules
    for filename, capsule in capsules:
        filepath = output_dir / filename
        print(f"Saving {filename}...")
        
        save_kpack(capsule, filepath)
        
        # Print capsule info
        original_size, compressed_size = capsule.calculate_size()
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        print(f"  Name: {capsule.metadata.name}")
        print(f"  Topic: {capsule.metadata.topic}")
        print(f"  Embeddings: {len(capsule.embeddings)}")
        print(f"  Sketches: {len(capsule.sketches)}")
        print(f"  Original size: {original_size / 1024:.1f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"  Compression ratio: {compression_ratio:.3f}")
        print(f"  Saved to: {filepath}")
        print()
    
    print(f"All example capsules saved to {output_dir}")
    print("\nTo load and use these capsules:")
    print("```python")
    print("from core_nn.memory.kpack import load_kpack")
    print("capsule = load_kpack('examples/kpacks/ml_fundamentals.kpack')")
    print("print(f'Loaded: {capsule.metadata.name}')")
    print("```")


if __name__ == "__main__":
    main()
