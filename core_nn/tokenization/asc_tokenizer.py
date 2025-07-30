"""
Adaptive Semantic Chunking (ASC) Tokenizer for CORE-NN.

This tokenizer implements:
1. Hybrid unit representation (word/subword/char)
2. Context-aware token merging
3. Self-evolving vocabulary
4. System command handling
"""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import time
from collections import OrderedDict
import logging

from .config import ASCConfig, TokenizerState
from .vocabulary import DynamicVocabulary, TokenType
from .merging import ContextualMerger
from .utils import TokenizerUtils


class ASCTokenizer:
    """Adaptive Semantic Chunking Tokenizer."""
    
    def __init__(self, config: ASCConfig = None, vocab_path: Optional[Path] = None):
        """Initialize ASC tokenizer."""
        self.config = config or ASCConfig()
        self.config.validate()
        
        # Core components
        self.vocabulary = DynamicVocabulary(self.config)
        self.merger = ContextualMerger(self.config)
        self.state = TokenizerState()
        
        # Caching
        self.tokenization_cache: OrderedDict[str, List[int]] = OrderedDict()
        self.detokenization_cache: OrderedDict[Tuple[int, ...], str] = OrderedDict()
        
        # Load vocabulary if provided
        if vocab_path and vocab_path.exists():
            self.load_vocabulary(vocab_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if self.config.enable_logging:
            self.logger.setLevel(logging.INFO)
    
    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # Check cache first (if enabled)
        cache_key = f"{text}_{add_special_tokens}"
        if self.config.cache_size > 0 and cache_key in self.tokenization_cache:
            self.state.cache_hits += 1
            return self.tokenization_cache[cache_key]

        self.state.cache_misses += 1
        
        # Step 1: Check for system commands
        system_command, remaining_text = TokenizerUtils.extract_system_command(text)
        
        token_ids = []
        
        # Add BOS token if requested
        if add_special_tokens:
            token_ids.append(self.config.special_tokens['<bos>'])
        
        # Handle system command
        if system_command:
            system_id = self._encode_system_command(system_command)
            if system_id is not None:
                token_ids.append(system_id)
            text = remaining_text
        
        if text:
            # Step 2: Basic tokenization
            basic_tokens = TokenizerUtils.basic_tokenize(text)
            
            # Step 3: Dynamic lexicon lookup with fallbacks
            processed_tokens = self._process_tokens(basic_tokens)
            
            # Step 4: Contextual merging
            if self.config.enable_contextual_merging:
                processed_tokens = self.merger.merge_tokens(processed_tokens)
            
            # Step 5: Convert to IDs
            for token in processed_tokens:
                # Always add as candidate for learning (even if already in vocab)
                if len(token) > 1 and TokenizerUtils.is_likely_word(token):
                    self.vocabulary.add_candidate_token(token)

                token_id = self._token_to_id(token)
                if token_id is not None:
                    token_ids.append(token_id)
                else:
                    # Token needs character-level processing
                    for char in token:
                        char_id = self._token_to_id(char)
                        if char_id is not None:
                            token_ids.append(char_id)
        
        # Add EOS token if requested
        if add_special_tokens:
            token_ids.append(self.config.special_tokens['<eos>'])
        
        # Update cache
        self._update_tokenization_cache(cache_key, token_ids)
        
        # Update statistics
        self.state.total_tokenizations += 1
        self.state.total_tokens_processed += len(token_ids)
        
        # Periodic maintenance
        self._periodic_maintenance()
        
        return token_ids
    
    def detokenize(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Reconstructed text
        """
        if not token_ids:
            return ""
        
        # Check cache (if enabled)
        cache_key = tuple(token_ids)
        if self.config.cache_size > 0 and cache_key in self.detokenization_cache:
            return self.detokenization_cache[cache_key]
        
        tokens = []
        for token_id in token_ids:
            token = self.vocabulary.get_token(token_id)
            if token is None:
                # Check if it's a character token (offset by 1000)
                if token_id >= 1000 and token_id < 65536:
                    char_code = token_id - 1000
                    if 32 <= char_code <= 126:  # Printable ASCII
                        token = chr(char_code)
                    else:
                        token = '<unk>'
                else:
                    token = '<unk>'
            elif skip_special_tokens and token in self.config.special_tokens:
                continue
            tokens.append(token)
        
        # Merge subword tokens
        tokens = TokenizerUtils.merge_subword_tokens(tokens)
        
        # Join tokens
        text = ' '.join(tokens)
        
        # Clean up spacing around punctuation
        text = self._clean_detokenized_text(text)
        
        # Update cache
        self._update_detokenization_cache(cache_key, text)
        
        return text
    
    def _process_tokens(self, tokens: List[str]) -> List[str]:
        """Process tokens through dynamic lexicon lookup with fallbacks."""
        processed = []

        for token in tokens:
            # Clean token
            clean_token = TokenizerUtils.clean_token(token)
            if not TokenizerUtils.validate_token(clean_token):
                continue

            # Try direct lookup first
            if self.vocabulary.get_token_id(clean_token) is not None:
                processed.append(clean_token)
                continue

            # Add as candidate for dynamic vocabulary and check if promoted
            self.vocabulary.add_candidate_token(clean_token)

            # Check again after adding as candidate (might have been promoted)
            if self.vocabulary.get_token_id(clean_token) is not None:
                processed.append(clean_token)
                continue

            # For unknown tokens, keep them as-is for now (they'll be handled in _token_to_id)
            # This preserves the token for future promotion
            processed.append(clean_token)

        return processed
    
    def _apply_fallback_strategy(self, token: str) -> List[str]:
        """Apply fallback strategy for unknown tokens."""
        # Strategy 1: Subword tokenization
        if self.config.subword_fallback and TokenizerUtils.is_likely_word(token):
            subwords = TokenizerUtils.subword_tokenize(token)
            # Check if subwords exist in vocabulary
            valid_subwords = []
            for subword in subwords:
                if self.vocabulary.get_token_id(subword) is not None:
                    valid_subwords.append(subword)
                else:
                    self.vocabulary.add_candidate_token(subword)
            
            if valid_subwords:
                return valid_subwords
        
        # Strategy 2: Character fallback
        if self.config.char_fallback:
            chars = TokenizerUtils.char_tokenize(token)
            return chars
        
        # Strategy 3: Return as unknown token
        return ['<unk>']
    
    def _token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID, handling all cases."""
        # Direct vocabulary lookup
        token_id = self.vocabulary.get_token_id(token)
        if token_id is not None:
            return token_id

        # For unknown tokens, use fallback strategies
        if len(token) == 1:
            # Handle single characters with ASCII offset
            char_id = ord(token) + 1000  # Offset to avoid conflicts
            if char_id < 65536:  # Stay within reasonable range
                return char_id
        elif TokenizerUtils.is_likely_word(token):
            # For word-like tokens, break into characters as fallback
            # This ensures we don't lose information
            return None  # Signal that this token needs character-level processing

        # Return unknown token ID as last resort
        return self.config.special_tokens['<unk>']
    
    def _encode_system_command(self, command: str) -> Optional[int]:
        """Encode system command to token ID."""
        if command in self.config.system_prefixes:
            # Check if already in vocabulary
            token_id = self.vocabulary.get_token_id(command)
            if token_id is not None:
                return token_id
            
            # Add to vocabulary as system token
            token_id = self.vocabulary.next_id
            self.vocabulary.next_id += 1
            self.vocabulary._add_token(command, token_id, TokenType.SYSTEM, frequency=1)
            return token_id
        
        return None
    
    def _clean_detokenized_text(self, text: str) -> str:
        """Clean up detokenized text."""
        # Remove spaces before punctuation
        import re
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Remove spaces around quotes
        text = re.sub(r'\s*"\s*([^"]*)\s*"\s*', r' "\1" ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _update_tokenization_cache(self, key: str, token_ids: List[int]) -> None:
        """Update tokenization cache with LRU eviction."""
        if self.config.cache_size <= 0:
            return  # Caching disabled

        if len(self.tokenization_cache) >= self.config.cache_size:
            self.tokenization_cache.popitem(last=False)  # Remove oldest
        self.tokenization_cache[key] = token_ids
    
    def _update_detokenization_cache(self, key: Tuple[int, ...], text: str) -> None:
        """Update detokenization cache with LRU eviction."""
        if self.config.cache_size <= 0:
            return  # Caching disabled

        if len(self.detokenization_cache) >= self.config.cache_size:
            self.detokenization_cache.popitem(last=False)  # Remove oldest
        self.detokenization_cache[key] = text
    
    def _periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        # Memory consolidation
        if (self.state.total_tokenizations - self.state.last_consolidation >= 
            self.config.memory_consolidation_interval):
            self.vocabulary.consolidate_memory()
            self.merger.clear_low_frequency_candidates()
            self.state.last_consolidation = self.state.total_tokenizations
        
        # Auto-save vocabulary
        if (self.config.save_dynamic_vocab and 
            self.state.total_tokenizations - self.state.last_save >= 
            self.config.auto_save_interval):
            self.save_vocabulary(Path(self.config.vocab_save_path))
            self.state.last_save = self.state.total_tokenizations
    
    def train_on_corpus(self, texts: List[str]) -> None:
        """Train tokenizer on a corpus of texts."""
        self.logger.info(f"Training tokenizer on {len(texts)} texts")
        
        # Collect token sequences for merger training
        token_sequences = []
        
        for text in texts:
            # Basic tokenization
            tokens = TokenizerUtils.basic_tokenize(text)
            token_sequences.append(tokens)
            
            # Process tokens to build vocabulary
            for token in tokens:
                clean_token = TokenizerUtils.clean_token(token)
                if TokenizerUtils.validate_token(clean_token):
                    self.vocabulary.add_candidate_token(clean_token)
        
        # Train contextual merger
        if self.config.enable_contextual_merging:
            self.merger.train_on_sequences(token_sequences)
        
        self.logger.info("Training completed")
    
    def add_tokens(self, tokens: List[str]) -> None:
        """Add tokens to vocabulary manually."""
        for token in tokens:
            clean_token = TokenizerUtils.clean_token(token)
            if TokenizerUtils.validate_token(clean_token):
                # Force addition to dynamic vocabulary
                for _ in range(self.config.adaptation_threshold):
                    self.vocabulary.add_candidate_token(clean_token)
    
    def save_vocabulary(self, path: Path) -> None:
        """Save vocabulary to file."""
        self.vocabulary.save_vocabulary(path)
        self.logger.info(f"Vocabulary saved to {path}")
    
    def load_vocabulary(self, path: Path) -> None:
        """Load vocabulary from file."""
        self.vocabulary.load_vocabulary(path)
        self.logger.info(f"Vocabulary loaded from {path}")
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocabulary.token_to_id)
    
    def get_stats(self) -> Dict:
        """Get comprehensive tokenizer statistics."""
        return {
            'tokenizer_state': self.state.get_stats(),
            'vocabulary_stats': self.vocabulary.get_vocabulary_stats(),
            'merger_stats': self.merger.get_merge_stats(),
            'config': {
                'vocab_size': self.get_vocab_size(),
                'max_vocab_size': self.config.max_vocab_size,
                'dynamic_vocab_size': self.config.dynamic_vocab_size,
                'enable_contextual_merging': self.config.enable_contextual_merging
            }
        }
