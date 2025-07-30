"""
Configuration system for Adaptive Semantic Chunking (ASC) Tokenizer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path
import json
import yaml


@dataclass
class ASCConfig:
    """Configuration for Adaptive Semantic Chunking Tokenizer."""
    
    # Core vocabulary settings
    base_vocab_size: int = 32000  # Base vocabulary size
    max_vocab_size: int = 50000   # Maximum vocabulary including dynamic tokens
    min_token_freq: int = 3       # Minimum frequency for token inclusion
    
    # Dynamic vocabulary settings
    dynamic_vocab_size: int = 8000    # Size of dynamic vocabulary cache
    adaptation_threshold: int = 5     # Uses before a token becomes permanent
    decay_factor: float = 0.95        # Decay factor for token frequencies
    
    # Contextual merging settings
    enable_contextual_merging: bool = True
    max_merge_length: int = 4         # Maximum tokens to merge into one
    merge_threshold: float = 0.7      # Threshold for merging decision
    ngram_order: int = 3              # N-gram model order for merging
    
    # Hybrid unit representation
    word_level_threshold: int = 100   # Frequency threshold for word-level tokens
    subword_fallback: bool = True     # Enable subword fallback
    char_fallback: bool = True        # Enable character fallback
    
    # System command prefixes
    system_prefixes: Set[str] = field(default_factory=lambda: {
        '#remember', '#recall', '#forget', '#define', '#compress',
        '#stats', '#help', '#save', '#load', '#clear'
    })
    
    # Special tokens
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        '<pad>': 0,
        '<unk>': 1, 
        '<bos>': 2,
        '<eos>': 3,
        '<mask>': 4,
        '<sep>': 5
    })
    
    # Runtime adaptation settings
    enable_runtime_adaptation: bool = True
    adaptation_window: int = 1000     # Window size for adaptation decisions
    memory_consolidation_interval: int = 500  # Steps between memory consolidation
    
    # Performance settings
    max_sequence_length: int = 2048
    batch_processing: bool = True
    parallel_processing: bool = False
    cache_size: int = 10000           # LRU cache size for tokenization results
    
    # Persistence settings
    save_dynamic_vocab: bool = True
    vocab_save_path: str = "tokenizer_vocab.json"
    auto_save_interval: int = 1000    # Auto-save interval in tokenizations
    
    # Debugging and monitoring
    enable_logging: bool = False
    log_adaptation_events: bool = False
    track_token_usage: bool = True
    
    def save_config(self, path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'base_vocab_size': self.base_vocab_size,
            'max_vocab_size': self.max_vocab_size,
            'min_token_freq': self.min_token_freq,
            'dynamic_vocab_size': self.dynamic_vocab_size,
            'adaptation_threshold': self.adaptation_threshold,
            'decay_factor': self.decay_factor,
            'enable_contextual_merging': self.enable_contextual_merging,
            'max_merge_length': self.max_merge_length,
            'merge_threshold': self.merge_threshold,
            'ngram_order': self.ngram_order,
            'word_level_threshold': self.word_level_threshold,
            'subword_fallback': self.subword_fallback,
            'char_fallback': self.char_fallback,
            'system_prefixes': list(self.system_prefixes),
            'special_tokens': self.special_tokens,
            'enable_runtime_adaptation': self.enable_runtime_adaptation,
            'adaptation_window': self.adaptation_window,
            'memory_consolidation_interval': self.memory_consolidation_interval,
            'max_sequence_length': self.max_sequence_length,
            'batch_processing': self.batch_processing,
            'parallel_processing': self.parallel_processing,
            'cache_size': self.cache_size,
            'save_dynamic_vocab': self.save_dynamic_vocab,
            'vocab_save_path': self.vocab_save_path,
            'auto_save_interval': self.auto_save_interval,
            'enable_logging': self.enable_logging,
            'log_adaptation_events': self.log_adaptation_events,
            'track_token_usage': self.track_token_usage
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, path: Path) -> 'ASCConfig':
        """Load configuration from JSON or YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)

        # Convert system_prefixes back to set
        if 'system_prefixes' in config_dict:
            if isinstance(config_dict['system_prefixes'], list):
                config_dict['system_prefixes'] = set(config_dict['system_prefixes'])

        return cls(**config_dict)

    @classmethod
    def load_preset(cls, preset_name: str) -> 'ASCConfig':
        """Load a preset configuration."""
        preset_path = Path(__file__).parent.parent.parent / "configs" / "tokenizer" / f"asc_{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset configuration not found: {preset_path}")
        return cls.load_config(preset_path)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_vocab_size <= self.base_vocab_size:
            raise ValueError("max_vocab_size must be greater than base_vocab_size")
        
        if self.dynamic_vocab_size <= 0:
            raise ValueError("dynamic_vocab_size must be positive")
        
        if not 0 < self.decay_factor <= 1:
            raise ValueError("decay_factor must be between 0 and 1")
        
        if not 0 < self.merge_threshold <= 1:
            raise ValueError("merge_threshold must be between 0 and 1")
        
        if self.max_merge_length < 2:
            raise ValueError("max_merge_length must be at least 2")
        
        if self.ngram_order < 1:
            raise ValueError("ngram_order must be at least 1")


@dataclass 
class TokenizerState:
    """Runtime state of the ASC tokenizer."""
    
    total_tokenizations: int = 0
    total_tokens_processed: int = 0
    dynamic_tokens_added: int = 0
    merges_performed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_consolidation: int = 0
    last_save: int = 0
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.total_tokenizations = 0
        self.total_tokens_processed = 0
        self.dynamic_tokens_added = 0
        self.merges_performed = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get tokenizer statistics."""
        return {
            'total_tokenizations': self.total_tokenizations,
            'total_tokens_processed': self.total_tokens_processed,
            'avg_tokens_per_input': (
                self.total_tokens_processed / max(1, self.total_tokenizations)
            ),
            'dynamic_tokens_added': self.dynamic_tokens_added,
            'merges_performed': self.merges_performed,
            'cache_hit_rate': (
                self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            )
        }
