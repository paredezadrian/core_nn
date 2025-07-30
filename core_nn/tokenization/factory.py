"""
Tokenizer factory for CORE-NN.
"""

from typing import Union, Optional
from pathlib import Path

from .asc_tokenizer import ASCTokenizer
from .config import ASCConfig
from ..config.schema import TokenizerConfig


class SimpleTokenizer:
    """Simple fallback tokenizer (for compatibility)."""
    
    def __init__(self):
        self.vocab_size = 1000
    
    def tokenize(self, text: str, add_special_tokens: bool = True) -> list:
        """Simple character-based tokenization."""
        tokens = [min(ord(c), 999) for c in text[:50]]
        if add_special_tokens:
            tokens = [2] + tokens + [3]  # BOS + tokens + EOS
        return tokens + [0] * (50 - len(tokens))  # Pad to 50
    
    def detokenize(self, token_ids: list, skip_special_tokens: bool = True) -> str:
        """Simple detokenization."""
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [0, 1, 2, 3]:  # Skip special tokens
                continue
            if token_id > 0:
                chars.append(chr(min(max(token_id, 32), 126)))
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_stats(self) -> dict:
        return {"type": "simple", "vocab_size": self.vocab_size}


class TokenizerFactory:
    """Factory for creating tokenizers."""
    
    @staticmethod
    def create_tokenizer(config: TokenizerConfig, 
                        vocab_path: Optional[Path] = None) -> Union[ASCTokenizer, SimpleTokenizer]:
        """
        Create a tokenizer based on configuration.
        
        Args:
            config: Tokenizer configuration
            vocab_path: Optional path to vocabulary file
            
        Returns:
            Tokenizer instance
        """
        if config.type.lower() == "asc":
            return TokenizerFactory._create_asc_tokenizer(config, vocab_path)
        elif config.type.lower() == "simple":
            return SimpleTokenizer()
        else:
            raise ValueError(f"Unknown tokenizer type: {config.type}")
    
    @staticmethod
    def _create_asc_tokenizer(config: TokenizerConfig, 
                             vocab_path: Optional[Path] = None) -> ASCTokenizer:
        """Create ASC tokenizer with configuration."""
        
        # Load base configuration
        if config.custom_config_path:
            # Load custom configuration
            asc_config = ASCConfig.load_config(Path(config.custom_config_path))
        else:
            # Load preset configuration
            try:
                asc_config = ASCConfig.load_preset(config.preset)
            except FileNotFoundError:
                # Fallback to default configuration
                asc_config = ASCConfig()
        
        # Apply overrides
        if config.overrides:
            for key, value in config.overrides.items():
                if hasattr(asc_config, key):
                    setattr(asc_config, key, value)
        
        # Create tokenizer
        return ASCTokenizer(asc_config, vocab_path)
    
    @staticmethod
    def get_available_tokenizers() -> list:
        """Get list of available tokenizer types."""
        return ["asc", "simple"]
    
    @staticmethod
    def get_available_presets() -> list:
        """Get list of available ASC presets."""
        return ["default", "edge", "research"]


def create_tokenizer_from_config(config: TokenizerConfig, 
                                vocab_path: Optional[Path] = None) -> Union[ASCTokenizer, SimpleTokenizer]:
    """
    Convenience function to create tokenizer from configuration.
    
    Args:
        config: Tokenizer configuration
        vocab_path: Optional path to vocabulary file
        
    Returns:
        Tokenizer instance
    """
    return TokenizerFactory.create_tokenizer(config, vocab_path)
