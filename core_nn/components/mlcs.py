"""
Multi-Level Compression Synthesizer (MLCS) - Knowledge compression and .kpack management.

The MLCS compresses large knowledge chunks into lightweight latent codes that can be
stored in .kpack files and swapped in/out dynamically like plugin memory cards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import gzip
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import hashlib

from ..config.schema import MLCSConfig
from ..memory.kpack import (
    KnowledgeCapsule, KnowledgeSketch, SketchType, CompressionType,
    load_kpack_for_mlcs, save_kpack_from_mlcs, create_capsule
)


@dataclass
class KnowledgePack:
    """Knowledge pack (.kpack) data structure."""
    pack_id: str
    name: str
    description: str
    latent_codes: torch.Tensor
    codebook_indices: torch.Tensor
    metadata: Dict[str, Any]
    compression_level: int
    original_size: int
    compressed_size: int
    creation_timestamp: float
    access_count: int = 0
    last_access: float = 0.0


class VectorQuantizer(nn.Module):
    """Vector quantization layer for compression."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vector quantization forward pass.
        
        Args:
            inputs: [batch_size, ..., embedding_dim]
            
        Returns:
            quantized: Quantized vectors
            loss: VQ loss
            indices: Codebook indices
        """
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(input_shape[:-1])


class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder for multi-level compression."""
    
    def __init__(self, input_dim: int, latent_dim: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers for each compression level
        self.encoders = nn.ModuleList()
        current_dim = input_dim
        
        for level in range(num_levels):
            # Compression ratio increases with level
            compression_ratio = 2 ** (level + 1)
            output_dim = max(latent_dim // compression_ratio, latent_dim // 8)
            
            encoder = nn.Sequential(
                nn.Linear(current_dim, current_dim // 2),
                nn.ReLU(),
                nn.Linear(current_dim // 2, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            self.encoders.append(encoder)
            current_dim = output_dim
    
    def forward(self, x: torch.Tensor, target_level: int = -1) -> List[torch.Tensor]:
        """Encode input to multiple compression levels."""
        if target_level == -1:
            target_level = self.num_levels
        
        encoded_levels = []
        current = x
        
        for level in range(min(target_level, self.num_levels)):
            current = self.encoders[level](current)
            encoded_levels.append(current)
        
        return encoded_levels


class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder for multi-level decompression."""
    
    def __init__(self, output_dim: int, latent_dim: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Decoder layers for each compression level
        self.decoders = nn.ModuleList()
        
        for level in range(num_levels):
            # Input dimension should match the quantizer output for this level
            input_dim = max(latent_dim // (2**(level+1)), latent_dim // 8)

            if level == num_levels - 1:
                next_dim = output_dim
            else:
                # Next dimension should match the quantizer output for the next level
                next_dim = max(latent_dim // (2**(level+2)), latent_dim // 8)
            
            decoder = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Linear(input_dim * 2, next_dim),
                nn.LayerNorm(next_dim) if level < num_levels - 1 else nn.Identity()
            )
            
            self.decoders.append(decoder)
    
    def forward(self, encoded_levels: List[torch.Tensor], from_level: int = 0) -> torch.Tensor:
        """Decode from specified compression level."""
        if from_level >= len(encoded_levels):
            raise ValueError(f"from_level {from_level} >= available levels {len(encoded_levels)}")
        
        current = encoded_levels[from_level]
        
        for level in range(from_level, self.num_levels):
            current = self.decoders[level](current)
        
        return current


class MultiLevelCompressionSynthesizer(nn.Module):
    """
    Multi-Level Compression Synthesizer (MLCS) - Knowledge compression and .kpack management.
    
    Compresses large knowledge chunks into lightweight latent codes that can be
    stored in .kpack files and swapped in/out dynamically.
    """
    
    def __init__(self, config: MLCSConfig, input_dim: int = 768):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = config.latent_dim
        self.num_levels = config.num_compression_levels
        self.codebook_size = config.codebook_size
        
        # Hierarchical encoder/decoder
        self.encoder = HierarchicalEncoder(input_dim, config.latent_dim, config.num_compression_levels)
        self.decoder = HierarchicalDecoder(input_dim, config.latent_dim, config.num_compression_levels)
        
        # Vector quantizers for each level
        self.quantizers = nn.ModuleList([
            VectorQuantizer(config.codebook_size, max(config.latent_dim // (2**(i+1)), config.latent_dim // 8))
            for i in range(config.num_compression_levels)
        ])
        
        # Knowledge pack storage
        self.loaded_kpacks: Dict[str, KnowledgePack] = {}
        self.kpack_directory = Path("kpacks")
        self.kpack_directory.mkdir(exist_ok=True)
        
        # Memory management
        self.memory_usage = 0
        self.max_memory_mb = config.kpack_max_size_mb * 10  # Allow 10 packs in memory
        
    def compress_knowledge(self, 
                          knowledge_data: torch.Tensor,
                          compression_level: int = -1,
                          name: str = "unnamed",
                          description: str = "") -> KnowledgePack:
        """
        Compress knowledge data into a knowledge pack.
        
        Args:
            knowledge_data: Input knowledge tensor [batch_size, input_dim]
            compression_level: Target compression level (-1 for maximum)
            name: Name for the knowledge pack
            description: Description of the knowledge
            
        Returns:
            KnowledgePack: Compressed knowledge pack
        """
        if compression_level == -1:
            compression_level = self.num_levels
        
        # Encode to multiple levels
        encoded_levels = self.encoder(knowledge_data, compression_level)
        
        # Quantize each level
        quantized_levels = []
        vq_losses = []
        indices_levels = []
        
        for i, encoded in enumerate(encoded_levels):
            if i < len(self.quantizers):
                quantized, vq_loss, indices = self.quantizers[i](encoded)
                quantized_levels.append(quantized)
                vq_losses.append(vq_loss)
                indices_levels.append(indices)
        
        # Create knowledge pack
        pack_id = self._generate_pack_id(name, knowledge_data)
        
        # Calculate sizes
        original_size = knowledge_data.numel() * 4  # Assuming float32
        compressed_size = sum(indices.numel() for indices in indices_levels) * 2  # Assuming int16
        
        # Store latent codes as a list since they have different dimensions
        # We'll pad them to the same size for stacking
        max_dim = max(q.shape[-1] for q in quantized_levels)
        padded_quantized = []
        padded_indices = []

        for quantized, indices in zip(quantized_levels, indices_levels):
            # Pad quantized tensor to max dimension
            if quantized.shape[-1] < max_dim:
                padding = max_dim - quantized.shape[-1]
                quantized_padded = F.pad(quantized, (0, padding), mode='constant', value=0)
            else:
                quantized_padded = quantized
            padded_quantized.append(quantized_padded)

            # Convert indices to float for stacking
            padded_indices.append(indices.float())

        kpack = KnowledgePack(
            pack_id=pack_id,
            name=name,
            description=description,
            latent_codes=torch.stack(padded_quantized),
            codebook_indices=torch.stack(padded_indices),
            metadata={
                "compression_level": compression_level,
                "vq_losses": [loss.item() for loss in vq_losses],
                "input_shape": list(knowledge_data.shape),
                "compression_ratio": original_size / compressed_size,
                "original_dimensions": [q.shape[-1] for q in quantized_levels]  # Store original dims
            },
            compression_level=compression_level,
            original_size=original_size,
            compressed_size=compressed_size,
            creation_timestamp=torch.tensor(0.0).item(),  # Simplified timestamp
        )
        
        # Store in loaded kpacks
        self.loaded_kpacks[kpack.pack_id] = kpack

        # Manage memory
        self.manage_memory()

        return kpack
    
    def decompress_knowledge(self, kpack: KnowledgePack, from_level: int = 0) -> torch.Tensor:
        """
        Decompress knowledge from a knowledge pack.

        Args:
            kpack: Knowledge pack to decompress
            from_level: Compression level to decompress from

        Returns:
            Decompressed knowledge tensor
        """
        # Extract latent codes
        latent_codes = kpack.latent_codes

        # Restore original dimensions if they were padded
        original_dims = kpack.metadata.get("original_dimensions")
        encoded_levels = []

        for i in range(latent_codes.size(0)):
            level_codes = latent_codes[i]

            # Unpad if necessary
            if original_dims and i < len(original_dims):
                original_dim = original_dims[i]
                if level_codes.shape[-1] > original_dim:
                    level_codes = level_codes[..., :original_dim]

            encoded_levels.append(level_codes)

        # Decode
        reconstructed = self.decoder(encoded_levels, from_level)

        # Update access statistics
        kpack.access_count += 1
        kpack.last_access = torch.tensor(0.0).item()  # Simplified timestamp
        
        return reconstructed
    
    def save_kpack(self, kpack: KnowledgePack, filepath: Optional[str] = None) -> str:
        """Save knowledge pack to disk."""
        if filepath is None:
            filepath_obj = self.kpack_directory / f"{kpack.pack_id}.kpack"
        else:
            filepath_obj = Path(filepath)
        
        # Prepare data for serialization
        kpack_data = {
            "pack_id": kpack.pack_id,
            "name": kpack.name,
            "description": kpack.description,
            "latent_codes": kpack.latent_codes.detach().cpu().numpy(),
            "codebook_indices": kpack.codebook_indices.detach().cpu().numpy(),
            "metadata": kpack.metadata,
            "compression_level": kpack.compression_level,
            "original_size": kpack.original_size,
            "compressed_size": kpack.compressed_size,
            "creation_timestamp": kpack.creation_timestamp,
            "access_count": kpack.access_count,
            "last_access": kpack.last_access
        }
        
        # Compress and save
        with gzip.open(filepath_obj, 'wb') as f:
            pickle.dump(kpack_data, f)

        return str(filepath_obj)
    
    def load_kpack(self, filepath: str) -> KnowledgePack:
        """Load knowledge pack from disk."""
        filepath_obj = Path(filepath)

        with gzip.open(filepath_obj, 'rb') as f:
            kpack_data = pickle.load(f)
        
        # Reconstruct knowledge pack
        kpack = KnowledgePack(
            pack_id=kpack_data["pack_id"],
            name=kpack_data["name"],
            description=kpack_data["description"],
            latent_codes=torch.from_numpy(kpack_data["latent_codes"]),
            codebook_indices=torch.from_numpy(kpack_data["codebook_indices"]),
            metadata=kpack_data["metadata"],
            compression_level=kpack_data["compression_level"],
            original_size=kpack_data["original_size"],
            compressed_size=kpack_data["compressed_size"],
            creation_timestamp=kpack_data["creation_timestamp"],
            access_count=kpack_data.get("access_count", 0),
            last_access=kpack_data.get("last_access", 0.0)
        )
        
        return kpack
    
    def manage_memory(self):
        """Manage memory usage by offloading least recently used kpacks."""
        current_memory = sum(
            kpack.compressed_size for kpack in self.loaded_kpacks.values()
        ) / (1024 * 1024)  # Convert to MB
        
        if current_memory > self.max_memory_mb:
            # Sort by last access time
            kpacks_by_access = sorted(
                self.loaded_kpacks.items(),
                key=lambda x: x[1].last_access
            )
            
            # Offload oldest kpacks
            for pack_id, kpack in kpacks_by_access:
                if current_memory <= self.max_memory_mb * 0.8:  # Target 80% usage
                    break
                
                # Save to disk if not already saved
                self.save_kpack(kpack)
                
                # Remove from memory
                del self.loaded_kpacks[pack_id]
                current_memory -= kpack.compressed_size / (1024 * 1024)
    
    def _generate_pack_id(self, name: str, data: torch.Tensor) -> str:
        """Generate unique pack ID."""
        # Create hash from name and data
        hasher = hashlib.md5()
        hasher.update(name.encode())
        hasher.update(data.cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        total_original = sum(kpack.original_size for kpack in self.loaded_kpacks.values())
        total_compressed = sum(kpack.compressed_size for kpack in self.loaded_kpacks.values())
        
        return {
            "loaded_kpacks": len(self.loaded_kpacks),
            "total_original_size_mb": total_original / (1024 * 1024),
            "total_compressed_size_mb": total_compressed / (1024 * 1024),
            "overall_compression_ratio": total_original / total_compressed if total_compressed > 0 else 0,
            "memory_usage_mb": total_compressed / (1024 * 1024),
            "memory_limit_mb": self.max_memory_mb,
            "memory_utilization": (total_compressed / (1024 * 1024)) / self.max_memory_mb
        }

    # New .kpack format integration methods

    def load_knowledge_capsule(self, path: Union[str, Path]) -> str:
        """
        Load a knowledge capsule (.kpack) and inject into MLCS.

        Args:
            path: Path to the .kpack file

        Returns:
            Pack ID of the loaded capsule

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If capsule format is invalid
        """
        # Load capsule using new format
        capsule_data = load_kpack_for_mlcs(path)

        # Extract embeddings and convert to latent codes
        embeddings = capsule_data['embeddings']
        if not embeddings:
            raise ValueError("Capsule contains no embeddings")

        # Use the first embedding as primary latent codes
        primary_embedding_name = list(embeddings.keys())[0]
        primary_embedding = embeddings[primary_embedding_name]

        # Convert to torch tensor if needed
        if isinstance(primary_embedding, np.ndarray):
            latent_codes = torch.from_numpy(primary_embedding.copy()).float()
        else:
            latent_codes = primary_embedding.float()

        # Ensure proper shape for MLCS
        if latent_codes.dim() == 1:
            latent_codes = latent_codes.unsqueeze(0)

        # Ensure the last dimension matches input_dim
        if latent_codes.shape[-1] != self.input_dim:
            # If dimensions don't match, we need to adapt
            if latent_codes.shape[-1] > self.input_dim:
                # Truncate
                latent_codes = latent_codes[..., :self.input_dim]
            else:
                # Pad with zeros
                padding_size = self.input_dim - latent_codes.shape[-1]
                padding = torch.zeros(*latent_codes.shape[:-1], padding_size)
                latent_codes = torch.cat([latent_codes, padding], dim=-1)

        # Encode the latent codes through the hierarchical encoder
        with torch.no_grad():  # Disable gradients for inference
            encoded_levels = self.encoder(latent_codes)

            # Quantize the first level
            quantized_codes, vq_loss, codebook_indices = self.quantizers[0](encoded_levels[0])

            # Store the encoded levels for reconstruction
            latent_codes = quantized_codes.detach()

        # Create KnowledgePack from capsule data
        metadata = capsule_data['metadata']
        pack_id = metadata.get('capsule_id', self._generate_pack_id(metadata['name'], latent_codes))

        kpack = KnowledgePack(
            pack_id=pack_id,
            name=metadata['name'],
            description=metadata['description'],
            latent_codes=quantized_codes,
            codebook_indices=codebook_indices,
            metadata={
                **metadata,
                'source_format': 'knowledge_capsule',
                'original_embeddings': list(embeddings.keys()),
                'sketch_types': metadata.get('sketch_types', [])
            },
            compression_level=len(self.quantizers),
            original_size=metadata.get('original_size', latent_codes.numel() * 4),
            compressed_size=metadata.get('compressed_size', quantized_codes.numel() * 4),
            creation_timestamp=torch.tensor(0.0).item()  # Simplified
        )

        # Store in loaded kpacks
        self.loaded_kpacks[pack_id] = kpack

        # Manage memory
        self.manage_memory()

        return pack_id

    def save_knowledge_capsule(self, pack_id: str, path: Union[str, Path],
                              topic: str = "", description: str = "") -> None:
        """
        Save a loaded knowledge pack as a knowledge capsule (.kpack).

        Args:
            pack_id: ID of the pack to save
            path: Path to save the .kpack file
            topic: Topic category for the capsule
            description: Description override (uses pack description if empty)

        Raises:
            KeyError: If pack_id not found
            ValueError: If pack data is invalid
        """
        if pack_id not in self.loaded_kpacks:
            raise KeyError(f"Knowledge pack '{pack_id}' not found")

        kpack = self.loaded_kpacks[pack_id]

        # Reconstruct embeddings from quantized codes
        reconstructed = self.decompress_knowledge(kpack)

        # Create embeddings dictionary
        embeddings = {
            'primary': reconstructed,
            'quantized': kpack.latent_codes
        }

        # Create sketches from available data
        sketches_data = {}
        if 'attention' in kpack.metadata.get('sketch_types', []):
            # Create attention sketch (placeholder - would need actual attention data)
            sketches_data['attention'] = kpack.latent_codes.mean(dim=0, keepdim=True)

        # Prepare MLCS data for saving
        mlcs_data = {
            'embeddings': embeddings,
            'sketches': sketches_data,
            'compression_data': {
                'quantization_levels': [
                    {
                        'level': i,
                        'codebook_size': q.num_embeddings,
                        'embedding_dim': q.embedding_dim,
                        'quantization_error': 0.0  # Would need to calculate
                    }
                    for i, q in enumerate(self.quantizers)
                ],
                'codebook_indices': kpack.codebook_indices.tolist(),
                'compression_level': kpack.compression_level,
                'original_pack_id': pack_id
            }
        }

        # Use provided description or fall back to pack description
        final_description = description or kpack.description
        final_topic = topic or kpack.metadata.get('topic', 'general')

        # Save using new format
        save_kpack_from_mlcs(
            mlcs_data=mlcs_data,
            path=path,
            name=kpack.name,
            topic=final_topic,
            description=final_description
        )

    def export_learned_knowledge(self, topic: str, path: Union[str, Path],
                                name: str = "", description: str = "") -> None:
        """
        Export currently learned knowledge as a knowledge capsule.

        Args:
            topic: Topic category for the exported knowledge
            path: Path to save the .kpack file
            name: Name for the capsule (auto-generated if empty)
            description: Description for the capsule
        """
        if not self.loaded_kpacks:
            raise ValueError("No knowledge packs loaded to export")

        # Combine all loaded knowledge
        all_latent_codes = []
        all_metadata = []

        for pack_id, kpack in self.loaded_kpacks.items():
            all_latent_codes.append(kpack.latent_codes)
            all_metadata.append({
                'pack_id': pack_id,
                'name': kpack.name,
                'description': kpack.description,
                'access_count': kpack.access_count
            })

        # Concatenate all latent codes (ensure same dimensions)
        # Find the minimum feature dimension to standardize
        min_features = min(codes.shape[-1] for codes in all_latent_codes)

        standardized_codes = []
        for codes in all_latent_codes:
            # Ensure 2D
            if codes.dim() > 2:
                codes = codes.view(codes.shape[0], -1)

            # Truncate or pad to min_features
            if codes.shape[-1] > min_features:
                codes = codes[:, :min_features]
            elif codes.shape[-1] < min_features:
                padding = torch.zeros(codes.shape[0], min_features - codes.shape[-1])
                codes = torch.cat([codes, padding], dim=-1)

            standardized_codes.append(codes)

        combined_codes = torch.cat(standardized_codes, dim=0)

        # Create embeddings (only store tensor data)
        embeddings = {
            'combined_knowledge': combined_codes
        }

        # Add individual pack embeddings with proper names
        for pack, codes in zip(all_metadata, standardized_codes):
            embeddings[f"pack_{pack['pack_id']}"] = codes

        # Create export metadata
        export_name = name or f"exported_knowledge_{topic}"
        export_description = description or f"Exported knowledge from {len(self.loaded_kpacks)} packs on topic: {topic}"

        # Prepare MLCS data
        mlcs_data = {
            'embeddings': embeddings,
            'sketches': {},
            'compression_data': {
                'export_timestamp': torch.tensor(0.0).item(),
                'source_packs': all_metadata,
                'compression_stats': self.get_compression_stats()
            }
        }

        # Save as knowledge capsule
        save_kpack_from_mlcs(
            mlcs_data=mlcs_data,
            path=path,
            name=export_name,
            topic=topic,
            description=export_description
        )
