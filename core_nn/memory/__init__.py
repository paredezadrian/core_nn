"""
Memory systems for CORE-NN.

Provides episodic, semantic, and working memory implementations,
as well as knowledge capsule (.kpack) I/O operations.
"""

# Note: These would be implemented as separate modules
# For now, we'll use placeholder imports since the main memory
# functionality is integrated into the core components

from .kpack import (
    KnowledgeCapsule,
    KnowledgeSketch,
    CapsuleMetadata,
    load_kpack,
    save_kpack,
    validate_kpack,
    create_capsule,
    merge_capsules
)
from .episodic_store import EpisodicStore, MemoryEntry

__all__ = [
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "EpisodicStore",
    "MemoryEntry",
    "KnowledgeCapsule",
    "KnowledgeSketch",
    "CapsuleMetadata",
    "load_kpack",
    "save_kpack",
    "validate_kpack",
    "create_capsule",
    "merge_capsules"
]

# Placeholder classes - in a full implementation these would be separate modules
class EpisodicMemory:
    """Placeholder for episodic memory system."""
    pass

class SemanticMemory:
    """Placeholder for semantic memory system."""
    pass

class WorkingMemory:
    """Placeholder for working memory system."""
    pass
