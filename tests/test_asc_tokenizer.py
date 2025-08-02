"""
Tests for Adaptive Semantic Chunking (ASC) Tokenizer.
"""

import pytest
import tempfile
from pathlib import Path

from core_nn.tokenization import (
    ASCTokenizer, ASCConfig, TokenizerFactory,
    create_tokenizer_from_config, SimpleTokenizer, TokenizerUtils
)
from core_nn.config.schema import TokenizerConfig


class TestASCConfig:
    """Test ASC configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ASCConfig()
        assert config.base_vocab_size == 32000
        assert config.max_vocab_size == 50000
        assert config.enable_contextual_merging is True
        assert config.subword_fallback is True
        assert config.char_fallback is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ASCConfig()
        config.validate()  # Should not raise
        
        # Test invalid config
        config.max_vocab_size = 10000  # Less than base_vocab_size
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config = ASCConfig(base_vocab_size=16000, enable_contextual_merging=False)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            config.save_config(config_path)
            loaded_config = ASCConfig.load_config(config_path)
            
            assert loaded_config.base_vocab_size == 16000
            assert loaded_config.enable_contextual_merging is False
        finally:
            config_path.unlink()
    
    def test_preset_loading(self):
        """Test preset configuration loading."""
        # Test default preset
        config = ASCConfig.load_preset("default")
        assert config.base_vocab_size == 32000
        
        # Test edge preset
        config = ASCConfig.load_preset("edge")
        assert config.base_vocab_size == 16000
        
        # Test research preset
        config = ASCConfig.load_preset("research")
        assert config.base_vocab_size == 64000


class TestASCTokenizer:
    """Test ASC tokenizer functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a test tokenizer."""
        config = ASCConfig(
            base_vocab_size=1000,
            max_vocab_size=2000,
            dynamic_vocab_size=500,
            adaptation_threshold=3,  # Lower threshold for testing
            enable_contextual_merging=True,
            enable_logging=False,
            cache_size=0  # Disable caching for testing
        )
        return ASCTokenizer(config)
    
    def test_basic_tokenization(self, tokenizer):
        """Test basic tokenization functionality."""
        text = "Hello world!"
        token_ids = tokenizer.tokenize(text)
        
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)
        
        # Test detokenization
        reconstructed = tokenizer.detokenize(token_ids)
        assert isinstance(reconstructed, str)
    
    def test_system_commands(self, tokenizer):
        """Test system command handling."""
        # Test system command
        text = "#remember this is important"
        token_ids = tokenizer.tokenize(text)
        
        # Should contain system command token
        assert len(token_ids) > 0
        
        # Test detokenization
        reconstructed = tokenizer.detokenize(token_ids)
        assert "#remember" in reconstructed or "remember" in reconstructed
    
    def test_empty_input(self, tokenizer):
        """Test handling of empty input."""
        assert tokenizer.tokenize("") == []
        assert tokenizer.detokenize([]) == ""
    
    def test_special_tokens(self, tokenizer):
        """Test special token handling."""
        text = "Hello"
        token_ids = tokenizer.tokenize(text, add_special_tokens=True)
        
        # Should have BOS and EOS tokens
        assert token_ids[0] == tokenizer.config.special_tokens['<bos>']
        assert token_ids[-1] == tokenizer.config.special_tokens['<eos>']
    
    def test_dynamic_vocabulary(self, tokenizer):
        """Test dynamic vocabulary adaptation."""
        # Use a unique token multiple times
        unique_token = "LumaXcelerate"

        # Use token enough times to exceed adaptation threshold
        total_uses = tokenizer.config.adaptation_threshold + 2
        for i in range(total_uses):
            tokenizer.tokenize(unique_token)

            # Check if promoted after threshold
            if i >= tokenizer.config.adaptation_threshold - 1:
                token_id = tokenizer.vocabulary.get_token_id(unique_token)
                if token_id is not None:
                    break

        # Check if token was added to dynamic vocabulary
        token_id = tokenizer.vocabulary.get_token_id(unique_token)
        assert token_id is not None, f"Token {unique_token} should be in vocabulary after {total_uses} uses (threshold: {tokenizer.config.adaptation_threshold})"
    
    def test_contextual_merging(self, tokenizer):
        """Test contextual merging functionality."""
        # Add a merge rule
        tokenizer.merger.add_merge_rule(["machine", "learning"], "machine_learning")

        # Test merging
        text = "machine learning is important"
        token_ids = tokenizer.tokenize(text)
        reconstructed = tokenizer.detokenize(token_ids)

        # Should contain some text (either merged or original tokens)
        assert len(reconstructed.strip()) > 0
        # Basic check that tokenization/detokenization works
        assert isinstance(token_ids, list) and len(token_ids) > 0
    
    def test_fallback_strategies(self, tokenizer):
        """Test fallback strategies for unknown tokens."""
        # Test with unknown word
        unknown_word = "supercalifragilisticexpialidocious"
        token_ids = tokenizer.tokenize(unknown_word)

        assert len(token_ids) > 0

        # Should be able to detokenize (may be character-level)
        reconstructed = tokenizer.detokenize(token_ids)
        # At minimum should have some characters
        assert len(reconstructed.replace(' ', '')) > 0  # Remove spaces and check
    
    def test_caching(self, tokenizer):
        """Test tokenization caching."""
        # Create a tokenizer with caching enabled for this test
        config = ASCConfig(
            base_vocab_size=1000,
            max_vocab_size=2000,
            dynamic_vocab_size=500,
            adaptation_threshold=10,  # High threshold to prevent vocabulary changes
            enable_contextual_merging=False,  # Disable for consistency
            enable_logging=False,
            cache_size=1000  # Enable caching
        )
        cached_tokenizer = ASCTokenizer(config)

        text = "This is a test"

        # First tokenization
        token_ids1 = cached_tokenizer.tokenize(text)
        cache_misses1 = cached_tokenizer.state.cache_misses

        # Second tokenization (should hit cache)
        token_ids2 = cached_tokenizer.tokenize(text)
        cache_hits = cached_tokenizer.state.cache_hits

        assert token_ids1 == token_ids2
        assert cache_hits > 0

    def test_merge_subword_tokens_handles_suffixes(self):
        """Ensure subword tokens merge across boundaries."""
        tokens = ["token@@", "izer"]
        assert TokenizerUtils.merge_subword_tokens(tokens) == ["tokenizer"]

        tokens = ["multi@@", "@@part@@", "token"]
        assert TokenizerUtils.merge_subword_tokens(tokens) == ["multiparttoken"]
    
    def test_statistics(self, tokenizer):
        """Test tokenizer statistics."""
        # Perform some tokenizations
        texts = ["Hello world", "This is a test", "Another example"]
        for text in texts:
            tokenizer.tokenize(text)
        
        stats = tokenizer.get_stats()
        assert 'tokenizer_state' in stats
        assert 'vocabulary_stats' in stats
        assert 'merger_stats' in stats
        assert 'config' in stats
        
        assert stats['tokenizer_state']['total_tokenizations'] == len(texts)
    
    def test_vocabulary_persistence(self, tokenizer):
        """Test vocabulary save and load."""
        # Add some dynamic tokens
        for _ in range(tokenizer.config.adaptation_threshold):
            tokenizer.tokenize("TestToken123")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            vocab_path = Path(f.name)
        
        try:
            # Save vocabulary
            tokenizer.save_vocabulary(vocab_path)
            
            # Create new tokenizer and load vocabulary
            new_tokenizer = ASCTokenizer(tokenizer.config)
            new_tokenizer.load_vocabulary(vocab_path)
            
            # Check if dynamic token was preserved
            token_id = new_tokenizer.vocabulary.get_token_id("TestToken123")
            assert token_id is not None
        finally:
            vocab_path.unlink()


class TestTokenizerFactory:
    """Test tokenizer factory."""
    
    def test_create_asc_tokenizer(self):
        """Test ASC tokenizer creation."""
        config = TokenizerConfig(type="asc", preset="default")
        tokenizer = TokenizerFactory.create_tokenizer(config)
        
        assert isinstance(tokenizer, ASCTokenizer)
        assert tokenizer.config.base_vocab_size == 32000
    
    def test_create_simple_tokenizer(self):
        """Test simple tokenizer creation."""
        config = TokenizerConfig(type="simple")
        tokenizer = TokenizerFactory.create_tokenizer(config)
        
        assert isinstance(tokenizer, SimpleTokenizer)
    
    def test_create_with_overrides(self):
        """Test tokenizer creation with overrides."""
        config = TokenizerConfig(
            type="asc",
            preset="default",
            overrides={"base_vocab_size": 16000, "enable_contextual_merging": False}
        )
        tokenizer = TokenizerFactory.create_tokenizer(config)
        
        assert isinstance(tokenizer, ASCTokenizer)
        assert tokenizer.config.base_vocab_size == 16000
        assert tokenizer.config.enable_contextual_merging is False
    
    def test_unknown_tokenizer_type(self):
        """Test handling of unknown tokenizer type."""
        config = TokenizerConfig(type="unknown")
        
        with pytest.raises(ValueError):
            TokenizerFactory.create_tokenizer(config)
    
    def test_available_tokenizers(self):
        """Test getting available tokenizers."""
        tokenizers = TokenizerFactory.get_available_tokenizers()
        assert "asc" in tokenizers
        assert "simple" in tokenizers
    
    def test_available_presets(self):
        """Test getting available presets."""
        presets = TokenizerFactory.get_available_presets()
        assert "default" in presets
        assert "edge" in presets
        assert "research" in presets


class TestSimpleTokenizer:
    """Test simple tokenizer (fallback)."""
    
    def test_basic_functionality(self):
        """Test basic simple tokenizer functionality."""
        tokenizer = SimpleTokenizer()
        
        text = "Hello"
        tokens = tokenizer.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        reconstructed = tokenizer.detokenize(tokens)
        assert isinstance(reconstructed, str)
    
    def test_vocab_size(self):
        """Test vocabulary size."""
        tokenizer = SimpleTokenizer()
        assert tokenizer.get_vocab_size() == 1000
    
    def test_stats(self):
        """Test statistics."""
        tokenizer = SimpleTokenizer()
        stats = tokenizer.get_stats()
        assert stats["type"] == "simple"
        assert stats["vocab_size"] == 1000


class TestIntegration:
    """Integration tests."""
    
    def test_tokenizer_config_integration(self):
        """Test integration with CORE-NN configuration."""
        config = TokenizerConfig(type="asc", preset="edge")
        tokenizer = create_tokenizer_from_config(config)
        
        assert isinstance(tokenizer, ASCTokenizer)
        assert tokenizer.config.base_vocab_size == 16000  # Edge preset
    
    def test_end_to_end_tokenization(self):
        """Test end-to-end tokenization workflow."""
        config = TokenizerConfig(type="asc", preset="default")
        tokenizer = create_tokenizer_from_config(config)
        
        # Test various inputs
        test_cases = [
            "Hello world!",
            "#remember this is important",
            "The quick brown fox jumps over the lazy dog.",
            "machine learning artificial intelligence",
            "LumaXcelerate is a great product name",
            ""
        ]
        
        for text in test_cases:
            token_ids = tokenizer.tokenize(text)
            reconstructed = tokenizer.detokenize(token_ids)
            
            # Basic sanity checks
            assert isinstance(token_ids, list)
            assert isinstance(reconstructed, str)
            
            if text:  # Non-empty input
                assert len(token_ids) > 0
