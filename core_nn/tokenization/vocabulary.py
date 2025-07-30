"""
Dynamic vocabulary management for ASC Tokenizer.
"""

from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Counter
from collections import defaultdict, OrderedDict
import json
import hashlib
import time
from pathlib import Path


class TokenType(Enum):
    """Types of tokens in the vocabulary."""
    SPECIAL = "special"      # Special tokens like <pad>, <unk>
    WORD = "word"           # Word-level tokens
    SUBWORD = "subword"     # Subword units (BPE-like)
    CHAR = "char"           # Character fallback
    SYSTEM = "system"       # System command tokens
    DYNAMIC = "dynamic"     # Runtime-learned tokens
    MERGED = "merged"       # Contextually merged tokens


class TokenInfo:
    """Information about a token in the vocabulary."""
    
    def __init__(self, token: str, token_id: int, token_type: TokenType,
                 frequency: int = 0, last_seen: float = None):
        self.token = token
        self.token_id = token_id
        self.token_type = token_type
        self.frequency = frequency
        self.last_seen = last_seen or time.time()
        self.creation_time = time.time()
        self.usage_contexts: List[str] = []  # Store contexts where token appears
        
    def update_usage(self, context: str = "") -> None:
        """Update token usage statistics."""
        self.frequency += 1
        self.last_seen = time.time()
        if context and len(self.usage_contexts) < 10:  # Limit context storage
            self.usage_contexts.append(context)
    
    def decay_frequency(self, decay_factor: float) -> None:
        """Apply decay to token frequency."""
        self.frequency = max(1, int(self.frequency * decay_factor))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'token': self.token,
            'token_id': self.token_id,
            'token_type': self.token_type.value,
            'frequency': self.frequency,
            'last_seen': self.last_seen,
            'creation_time': self.creation_time,
            'usage_contexts': self.usage_contexts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TokenInfo':
        """Create from dictionary."""
        token_info = cls(
            token=data['token'],
            token_id=data['token_id'],
            token_type=TokenType(data['token_type']),
            frequency=data['frequency'],
            last_seen=data['last_seen']
        )
        token_info.creation_time = data.get('creation_time', time.time())
        token_info.usage_contexts = data.get('usage_contexts', [])
        return token_info


class DynamicVocabulary:
    """Dynamic vocabulary that adapts at runtime."""
    
    def __init__(self, config):
        self.config = config
        
        # Core vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_info: Dict[str, TokenInfo] = {}
        
        # Dynamic vocabulary management
        self.dynamic_tokens: OrderedDict[str, TokenInfo] = OrderedDict()
        self.candidate_tokens: Counter = Counter()
        
        # Token type tracking
        self.tokens_by_type: Dict[TokenType, Set[str]] = defaultdict(set)
        
        # Next available ID
        self.next_id = len(config.special_tokens)
        
        # Initialize with special tokens
        self._initialize_special_tokens()
        
        # Runtime statistics
        self.adaptation_events = 0
        self.consolidation_events = 0
    
    def _initialize_special_tokens(self) -> None:
        """Initialize vocabulary with special tokens."""
        for token, token_id in self.config.special_tokens.items():
            self._add_token(token, token_id, TokenType.SPECIAL)
            self.next_id = max(self.next_id, token_id + 1)
    
    def _add_token(self, token: str, token_id: int, token_type: TokenType,
                   frequency: int = 0) -> None:
        """Add a token to the vocabulary."""
        if token in self.token_to_id:
            return  # Token already exists
        
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        
        token_info = TokenInfo(token, token_id, token_type, frequency)
        self.token_info[token] = token_info
        self.tokens_by_type[token_type].add(token)
    
    def add_base_vocabulary(self, vocab: Dict[str, int]) -> None:
        """Add base vocabulary (word-level and subword tokens)."""
        for token, frequency in vocab.items():
            if token not in self.token_to_id:
                token_type = TokenType.WORD if frequency >= self.config.word_level_threshold else TokenType.SUBWORD
                self._add_token(token, self.next_id, token_type, frequency)
                self.next_id += 1
    
    def get_token_id(self, token: str, context: str = "") -> Optional[int]:
        """Get token ID, updating usage statistics."""
        if token in self.token_to_id:
            self.token_info[token].update_usage(context)
            return self.token_to_id[token]
        return None
    
    def get_token(self, token_id: int) -> Optional[str]:
        """Get token from ID."""
        return self.id_to_token.get(token_id)
    
    def add_candidate_token(self, token: str, context: str = "") -> None:
        """Add a candidate token for potential inclusion."""
        if token in self.token_to_id:
            return  # Already in vocabulary
        
        self.candidate_tokens[token] += 1
        
        # Check if token should be promoted to dynamic vocabulary
        if self.candidate_tokens[token] >= self.config.adaptation_threshold:
            self._promote_to_dynamic(token)
    
    def _promote_to_dynamic(self, token: str) -> None:
        """Promote a candidate token to dynamic vocabulary."""
        if len(self.dynamic_tokens) >= self.config.dynamic_vocab_size:
            self._evict_least_used_dynamic()
        
        token_id = self.next_id
        self.next_id += 1
        
        frequency = self.candidate_tokens[token]
        self._add_token(token, token_id, TokenType.DYNAMIC, frequency)
        
        token_info = self.token_info[token]
        self.dynamic_tokens[token] = token_info
        
        # Remove from candidates
        del self.candidate_tokens[token]
        self.adaptation_events += 1
    
    def _evict_least_used_dynamic(self) -> None:
        """Evict the least used dynamic token."""
        if not self.dynamic_tokens:
            return
        
        # Find least recently used token
        oldest_token = min(self.dynamic_tokens.keys(),
                          key=lambda t: self.token_info[t].last_seen)
        
        # Remove from all mappings
        token_info = self.dynamic_tokens[oldest_token]
        token_id = token_info.token_id
        
        del self.token_to_id[oldest_token]
        del self.id_to_token[token_id]
        del self.token_info[oldest_token]
        del self.dynamic_tokens[oldest_token]
        self.tokens_by_type[TokenType.DYNAMIC].discard(oldest_token)
    
    def add_merged_token(self, tokens: List[str], merged_token: str) -> int:
        """Add a contextually merged token."""
        if merged_token in self.token_to_id:
            return self.token_to_id[merged_token]
        
        token_id = self.next_id
        self.next_id += 1
        
        # Calculate frequency as minimum of component tokens
        min_freq = min(self.token_info[t].frequency for t in tokens if t in self.token_info)
        
        self._add_token(merged_token, token_id, TokenType.MERGED, min_freq)
        return token_id
    
    def consolidate_memory(self) -> None:
        """Consolidate vocabulary memory by applying decay and cleanup."""
        # Apply decay to all token frequencies
        for token_info in self.token_info.values():
            if token_info.token_type in [TokenType.DYNAMIC, TokenType.MERGED]:
                token_info.decay_frequency(self.config.decay_factor)
        
        # Remove very low frequency dynamic tokens
        to_remove = []
        for token, token_info in self.dynamic_tokens.items():
            if token_info.frequency < 2:  # Very low frequency
                to_remove.append(token)
        
        for token in to_remove:
            self._evict_token(token)
        
        self.consolidation_events += 1
    
    def _evict_token(self, token: str) -> None:
        """Evict a specific token from vocabulary."""
        if token not in self.token_to_id:
            return
        
        token_info = self.token_info[token]
        token_id = token_info.token_id
        token_type = token_info.token_type
        
        del self.token_to_id[token]
        del self.id_to_token[token_id]
        del self.token_info[token]
        self.tokens_by_type[token_type].discard(token)
        
        if token in self.dynamic_tokens:
            del self.dynamic_tokens[token]
    
    def get_vocabulary_stats(self) -> Dict:
        """Get vocabulary statistics."""
        stats = {
            'total_tokens': len(self.token_to_id),
            'dynamic_tokens': len(self.dynamic_tokens),
            'candidate_tokens': len(self.candidate_tokens),
            'adaptation_events': self.adaptation_events,
            'consolidation_events': self.consolidation_events,
            'tokens_by_type': {
                token_type.value: len(tokens) 
                for token_type, tokens in self.tokens_by_type.items()
            }
        }
        return stats
    
    def save_vocabulary(self, path: Path) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'token_info': {
                token: info.to_dict() 
                for token, info in self.token_info.items()
            },
            'candidate_tokens': dict(self.candidate_tokens),
            'next_id': self.next_id,
            'adaptation_events': self.adaptation_events,
            'consolidation_events': self.consolidation_events
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load_vocabulary(self, path: Path) -> None:
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Clear existing vocabulary (except special tokens)
        self._clear_non_special_tokens()
        
        # Load token info
        for token, info_dict in vocab_data['token_info'].items():
            token_info = TokenInfo.from_dict(info_dict)
            if token_info.token_type != TokenType.SPECIAL:  # Don't overwrite special tokens
                self.token_to_id[token] = token_info.token_id
                self.id_to_token[token_info.token_id] = token
                self.token_info[token] = token_info
                self.tokens_by_type[token_info.token_type].add(token)
                
                if token_info.token_type == TokenType.DYNAMIC:
                    self.dynamic_tokens[token] = token_info
        
        # Load other data
        self.candidate_tokens = Counter(vocab_data.get('candidate_tokens', {}))
        self.next_id = vocab_data.get('next_id', self.next_id)
        self.adaptation_events = vocab_data.get('adaptation_events', 0)
        self.consolidation_events = vocab_data.get('consolidation_events', 0)
    
    def _clear_non_special_tokens(self) -> None:
        """Clear all non-special tokens from vocabulary."""
        special_tokens = set(self.config.special_tokens.keys())
        
        tokens_to_remove = [
            token for token in self.token_to_id.keys()
            if token not in special_tokens
        ]
        
        for token in tokens_to_remove:
            self._evict_token(token)
