"""
Utility functions for ASC Tokenizer.
"""

import re
import unicodedata
from typing import List, Set, Dict, Tuple, Optional
import hashlib


class TokenizerUtils:
    """Utility functions for tokenization."""
    
    # Common punctuation and special characters
    PUNCTUATION = set('.,!?;:()[]{}"\'-_/\\@#$%^&*+=<>|`~')
    
    # System command pattern
    SYSTEM_COMMAND_PATTERN = re.compile(r'^#\w+')
    
    # Word boundary patterns
    WORD_BOUNDARY_PATTERN = re.compile(r'\b')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    # Character type patterns
    LETTER_PATTERN = re.compile(r'[a-zA-Z]')
    DIGIT_PATTERN = re.compile(r'\d')
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize input text."""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters
        text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
        
        # Normalize whitespace
        text = TokenizerUtils.WHITESPACE_PATTERN.sub(' ', text)
        
        return text.strip()
    
    @staticmethod
    def basic_tokenize(text: str) -> List[str]:
        """Basic tokenization splitting on whitespace and punctuation."""
        # Normalize text first
        text = TokenizerUtils.normalize_text(text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Further split on punctuation
        refined_tokens = []
        for token in tokens:
            refined_tokens.extend(TokenizerUtils._split_punctuation(token))
        
        return [token for token in refined_tokens if token]
    
    @staticmethod
    def _split_punctuation(token: str) -> List[str]:
        """Split token on punctuation boundaries."""
        if not token:
            return []
        
        result = []
        current = ""
        
        for char in token:
            if char in TokenizerUtils.PUNCTUATION:
                if current:
                    result.append(current)
                    current = ""
                result.append(char)
            else:
                current += char
        
        if current:
            result.append(current)
        
        return result
    
    @staticmethod
    def is_system_command(text: str) -> bool:
        """Check if text is a system command."""
        return bool(TokenizerUtils.SYSTEM_COMMAND_PATTERN.match(text.strip()))
    
    @staticmethod
    def extract_system_command(text: str) -> Tuple[str, str]:
        """Extract system command and remaining text."""
        text = text.strip()
        match = TokenizerUtils.SYSTEM_COMMAND_PATTERN.match(text)
        
        if match:
            command = match.group(0)
            remaining = text[len(command):].strip()
            return command, remaining
        
        return "", text
    
    @staticmethod
    def subword_tokenize(word: str, max_length: int = 6) -> List[str]:
        """Simple subword tokenization using character n-grams."""
        if len(word) <= max_length:
            return [word]
        
        subwords = []
        
        # Add prefixes
        for i in range(1, min(max_length + 1, len(word))):
            subwords.append(word[:i] + "@@")
        
        # Add suffixes  
        for i in range(max(1, len(word) - max_length), len(word)):
            subwords.append("@@" + word[i:])
        
        # Add middle parts if word is long
        if len(word) > max_length * 2:
            for i in range(max_length, len(word) - max_length, max_length):
                end = min(i + max_length, len(word) - max_length)
                subwords.append("@@" + word[i:end] + "@@")
        
        return subwords
    
    @staticmethod
    def char_tokenize(text: str) -> List[str]:
        """Character-level tokenization."""
        return list(text)
    
    @staticmethod
    def generate_token_hash(token: str) -> str:
        """Generate a hash for a token (for fallback IDs)."""
        return hashlib.md5(token.encode('utf-8')).hexdigest()[:8]
    
    @staticmethod
    def is_likely_word(token: str) -> bool:
        """Check if token is likely a complete word."""
        if len(token) < 2:
            return False
        
        # Check if it's mostly letters
        letter_count = sum(1 for char in token if TokenizerUtils.LETTER_PATTERN.match(char))
        return letter_count / len(token) > 0.7
    
    @staticmethod
    def is_number(token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def get_token_features(token: str) -> Dict[str, bool]:
        """Extract features from a token."""
        return {
            'is_word': TokenizerUtils.is_likely_word(token),
            'is_number': TokenizerUtils.is_number(token),
            'has_punctuation': any(char in TokenizerUtils.PUNCTUATION for char in token),
            'is_capitalized': token and token[0].isupper(),
            'is_all_caps': token.isupper(),
            'has_digits': any(char.isdigit() for char in token),
            'length': len(token),
            'starts_with_hash': token.startswith('#'),
            'has_at_symbol': '@@' in token  # Subword marker
        }
    
    @staticmethod
    def clean_token(token: str) -> str:
        """Clean a token for processing."""
        # Remove excessive whitespace
        token = token.strip()
        
        # Remove control characters
        token = ''.join(char for char in token if not unicodedata.category(char).startswith('C'))
        
        return token
    
    @staticmethod
    def merge_subword_tokens(tokens: List[str]) -> List[str]:
        """Merge subword tokens back into words."""
        if not tokens:
            return []
        
        merged = []
        current_word = ""
        
        for token in tokens:
            if token.startswith("@@") and token.endswith("@@"):
                # Middle piece of a word
                current_word += token[2:-2]
            elif token.endswith("@@"):
                # Start or continue a word
                current_word += token[:-2]
            elif token.startswith("@@"):
                # End a word
                current_word += token[2:]
                if current_word:
                    merged.append(current_word)
                    current_word = ""
            else:
                # Final piece of a split word or a standalone token
                if current_word:
                    current_word += token
                    merged.append(current_word)
                    current_word = ""
                else:
                    merged.append(token)
        
        # Handle any remaining partial word
        if current_word:
            merged.append(current_word)
        
        return merged
    
    @staticmethod
    def calculate_compression_ratio(original_tokens: List[str], 
                                  compressed_tokens: List[str]) -> float:
        """Calculate compression ratio achieved by tokenization."""
        if not original_tokens:
            return 1.0
        
        return len(compressed_tokens) / len(original_tokens)
    
    @staticmethod
    def estimate_token_importance(token: str, frequency: int, 
                                context_diversity: int) -> float:
        """Estimate the importance of a token for vocabulary inclusion."""
        # Base importance from frequency
        freq_score = min(1.0, frequency / 100.0)
        
        # Bonus for context diversity
        diversity_score = min(1.0, context_diversity / 10.0)
        
        # Token type bonuses
        features = TokenizerUtils.get_token_features(token)
        type_bonus = 0.0
        
        if features['is_word']:
            type_bonus += 0.3
        if features['is_number']:
            type_bonus += 0.1
        if features['starts_with_hash']:
            type_bonus += 0.5  # System commands are important
        
        # Length penalty for very long tokens
        length_penalty = 1.0 if len(token) <= 10 else 0.8
        
        importance = (freq_score + diversity_score + type_bonus) * length_penalty
        return min(1.0, importance)
    
    @staticmethod
    def validate_token(token: str) -> bool:
        """Validate if a token is acceptable."""
        if not token or len(token) > 50:  # Too long
            return False
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for char in token if char in TokenizerUtils.PUNCTUATION) / len(token)
        if punct_ratio > 0.8:  # Too much punctuation
            return False
        
        # Check for control characters
        if any(unicodedata.category(char).startswith('C') for char in token):
            return False
        
        return True
