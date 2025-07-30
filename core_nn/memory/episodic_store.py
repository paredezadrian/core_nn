"""
Episodic Memory Store for CORE-NN.

Provides a unified interface for episodic memory operations across BCM and IGPM components.
Handles remember(), recall(), and forget() operations with optional disk persistence.
"""

import torch
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from ..components.bcm import BiologicalCoreMemory
from ..components.igpm import InstructionGuidedPlasticityModule, EpisodicMemory


@dataclass
class MemoryEntry:
    """Unified memory entry for episodic store."""
    term: str
    content: str
    timestamp: datetime
    source: str  # 'bcm', 'igpm', or 'both'
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[torch.Tensor] = None
    salience: float = 0.5
    usage_count: int = 0


class EpisodicStore:
    """
    Unified episodic memory store that manages memory operations across BCM and IGPM.
    
    Provides:
    - remember(term, content): Store new episodic memory
    - recall(term): Retrieve memories by term
    - forget(term): Remove memories by term
    - Optional disk persistence for session continuity
    """
    
    def __init__(self, 
                 bcm: BiologicalCoreMemory,
                 igpm: InstructionGuidedPlasticityModule,
                 cache_dir: Optional[Path] = None,
                 enable_disk_cache: bool = False):
        """
        Initialize episodic store.
        
        Args:
            bcm: BiologicalCoreMemory instance
            igpm: InstructionGuidedPlasticityModule instance
            cache_dir: Directory for disk cache (optional)
            enable_disk_cache: Whether to enable disk persistence
        """
        self.bcm = bcm
        self.igpm = igpm
        self.cache_dir = cache_dir
        self.enable_disk_cache = enable_disk_cache
        
        # In-memory store for quick access
        self.memory_entries: Dict[str, List[MemoryEntry]] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup disk cache if enabled
        if self.enable_disk_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def remember(self, term: str, content: str = None) -> Dict[str, Any]:
        """
        Store new episodic memory.
        
        Args:
            term: Memory term/key
            content: Memory content (if None, uses term as content)
            
        Returns:
            Dictionary with storage results
        """
        if content is None:
            content = term
        
        # Create memory entry
        entry = MemoryEntry(
            term=term,
            content=content,
            timestamp=datetime.now(),
            source='both',
            metadata={'explicit': True}
        )
        
        # Generate embedding for the content
        # Use a simple approach - in practice this could be more sophisticated
        embedding = self._generate_embedding(content)
        entry.embedding = embedding
        
        # Store in BCM
        bcm_result = self.bcm.remember_explicit(
            embedding, 
            {"instruction": content, "term": term, "explicit": True}
        )
        
        # Store in IGPM
        igpm_result = self.igpm.remember_explicit(content, embedding)
        
        # Store in local memory
        if term not in self.memory_entries:
            self.memory_entries[term] = []
        self.memory_entries[term].append(entry)
        
        # Save to disk if enabled
        if self.enable_disk_cache:
            self._save_to_disk(term, entry)
        
        result = {
            'term': term,
            'content': content,
            'bcm_stored': bcm_result is not None,
            'igpm_stored': igpm_result.get('memory_stored', False),
            'timestamp': entry.timestamp.isoformat(),
            'entry_id': id(entry)
        }
        
        self.logger.info(f"Remembered: {term} -> {content[:50]}...")
        return result
    
    def recall(self, term: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Recall memories by term.
        
        Args:
            term: Memory term to search for
            top_k: Maximum number of memories to return
            
        Returns:
            Dictionary with recalled memories
        """
        results = {
            'term': term,
            'local_memories': [],
            'bcm_memories': [],
            'igpm_memories': [],
            'total_found': 0
        }
        
        # Search local memory entries
        local_matches = []
        for stored_term, entries in self.memory_entries.items():
            if term.lower() in stored_term.lower():
                for entry in entries:
                    entry.usage_count += 1
                    local_matches.append(entry)
        
        # Sort by relevance (usage count and recency)
        local_matches.sort(key=lambda x: (x.usage_count, x.timestamp), reverse=True)
        results['local_memories'] = [self._entry_to_dict(entry) for entry in local_matches[:top_k]]
        
        # Recall from BCM
        try:
            bcm_memories = self.bcm.recall_by_context({"instruction": term}, top_k)
            results['bcm_memories'] = bcm_memories[:top_k] if bcm_memories else []
        except Exception as e:
            self.logger.warning(f"BCM recall failed: {e}")
            results['bcm_memories'] = []
        
        # Recall from IGPM
        try:
            igpm_memories = self.igpm.recall_by_instruction(term, top_k)
            results['igpm_memories'] = [
                {
                    'instruction': mem.instruction,
                    'timestamp': mem.timestamp,
                    'usage_count': mem.usage_count,
                    'success_score': mem.success_score
                }
                for mem in igpm_memories
            ]
        except Exception as e:
            self.logger.warning(f"IGPM recall failed: {e}")
            results['igpm_memories'] = []
        
        results['total_found'] = (
            len(results['local_memories']) + 
            len(results['bcm_memories']) + 
            len(results['igpm_memories'])
        )
        
        self.logger.info(f"Recalled {results['total_found']} memories for: {term}")
        return results
    
    def forget(self, term: str) -> Dict[str, Any]:
        """
        Forget memories related to term.
        
        Args:
            term: Memory term to forget
            
        Returns:
            Dictionary with forgetting results
        """
        results = {
            'term': term,
            'local_removed': 0,
            'bcm_removed': 0,
            'igpm_removed': 0,
            'disk_removed': 0
        }
        
        # Remove from local memory
        initial_local_count = sum(len(entries) for entries in self.memory_entries.values())
        
        # Remove matching entries
        terms_to_remove = []
        for stored_term in list(self.memory_entries.keys()):
            if term.lower() in stored_term.lower():
                results['local_removed'] += len(self.memory_entries[stored_term])
                terms_to_remove.append(stored_term)
        
        for term_to_remove in terms_to_remove:
            del self.memory_entries[term_to_remove]
        
        # Remove from BCM (simplified - in practice would need more sophisticated matching)
        initial_bcm_count = len(self.bcm.memory_slots)
        self.bcm.memory_slots = [
            slot for slot in self.bcm.memory_slots
            if not (slot.context_tags and 
                   any(term.lower() in str(v).lower() for v in slot.context_tags.values()))
        ]
        results['bcm_removed'] = initial_bcm_count - len(self.bcm.memory_slots)
        
        # Remove from IGPM
        initial_igpm_count = len(self.igpm.episodic_memories)
        self.igpm.episodic_memories = [
            memory for memory in self.igpm.episodic_memories
            if term.lower() not in memory.instruction.lower()
        ]
        results['igpm_removed'] = initial_igpm_count - len(self.igpm.episodic_memories)
        
        # Remove from disk cache
        if self.enable_disk_cache:
            results['disk_removed'] = self._remove_from_disk(term)
        
        self.logger.info(f"Forgot memories for: {term} (removed {sum(results[k] for k in results if k.endswith('_removed'))} total)")
        return results
    
    def _generate_embedding(self, text: str) -> torch.Tensor:
        """Generate embedding for text content."""
        # Simple embedding generation - in practice this could use the model's embedding layer
        # For now, create a random embedding based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        torch.manual_seed(seed)
        return torch.randn(1, self.bcm.embedding_dim)
    
    def _entry_to_dict(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Convert memory entry to dictionary."""
        result = asdict(entry)
        result['timestamp'] = entry.timestamp.isoformat()
        if entry.embedding is not None:
            result['embedding'] = entry.embedding.tolist()
        return result
    
    def _save_to_disk(self, term: str, entry: MemoryEntry):
        """Save memory entry to disk."""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"{term.replace(' ', '_')}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            self.logger.warning(f"Failed to save to disk: {e}")
    
    def _load_from_disk(self):
        """Load memory entries from disk."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    if entry.term not in self.memory_entries:
                        self.memory_entries[entry.term] = []
                    self.memory_entries[entry.term].append(entry)
        except Exception as e:
            self.logger.warning(f"Failed to load from disk: {e}")
    
    def _remove_from_disk(self, term: str) -> int:
        """Remove memory entries from disk."""
        if not self.cache_dir:
            return 0
        
        removed_count = 0
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                if term.lower() in cache_file.stem.lower():
                    cache_file.unlink()
                    removed_count += 1
        except Exception as e:
            self.logger.warning(f"Failed to remove from disk: {e}")
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episodic store statistics."""
        total_entries = sum(len(entries) for entries in self.memory_entries.values())
        
        return {
            'total_terms': len(self.memory_entries),
            'total_entries': total_entries,
            'bcm_memories': len(self.bcm.memory_slots),
            'igpm_memories': len(self.igpm.episodic_memories),
            'disk_cache_enabled': self.enable_disk_cache,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None
        }
