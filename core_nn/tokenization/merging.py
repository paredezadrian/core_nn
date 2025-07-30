"""
Contextual merging component for ASC Tokenizer.
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import math
import re
from dataclasses import dataclass


@dataclass
class MergeCandidate:
    """Candidate for token merging."""
    tokens: Tuple[str, ...]
    frequency: int
    contexts: List[str]
    score: float
    
    def __post_init__(self):
        self.merged_token = "_".join(self.tokens)


class NGramModel:
    """Simple n-gram language model for contextual merging decisions."""
    
    def __init__(self, order: int = 3):
        self.order = order
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.total_ngrams = 0
    
    def train(self, token_sequences: List[List[str]]) -> None:
        """Train the n-gram model on token sequences."""
        for sequence in token_sequences:
            # Add sentence boundaries
            padded_sequence = ['<s>'] * (self.order - 1) + sequence + ['</s>']
            
            # Extract n-grams
            for i in range(len(padded_sequence) - self.order + 1):
                ngram = tuple(padded_sequence[i:i + self.order])
                context = ngram[:-1]
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.total_ngrams += 1
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get probability of an n-gram."""
        if len(ngram) != self.order:
            return 0.0
        
        context = ngram[:-1]
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context]
        
        if context_count == 0:
            return 1.0 / len(self.ngram_counts)  # Uniform fallback
        
        # Add-one smoothing
        return (ngram_count + 1) / (context_count + len(self.ngram_counts))
    
    def get_perplexity(self, token_sequence: List[str]) -> float:
        """Calculate perplexity of a token sequence."""
        if len(token_sequence) < self.order:
            return float('inf')
        
        padded_sequence = ['<s>'] * (self.order - 1) + token_sequence + ['</s>']
        log_prob_sum = 0.0
        ngram_count = 0
        
        for i in range(len(padded_sequence) - self.order + 1):
            ngram = tuple(padded_sequence[i:i + self.order])
            prob = self.get_probability(ngram)
            if prob > 0:
                log_prob_sum += math.log(prob)
                ngram_count += 1
        
        if ngram_count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / ngram_count
        return math.exp(-avg_log_prob)
    
    def should_merge(self, tokens: List[str], threshold: float = 0.7) -> bool:
        """Determine if tokens should be merged based on perplexity."""
        if len(tokens) < 2:
            return False
        
        # Calculate perplexity with and without merging
        original_perplexity = self.get_perplexity(tokens)
        
        # Create merged version
        merged_tokens = ['_'.join(tokens)]
        merged_perplexity = self.get_perplexity(merged_tokens)
        
        # Lower perplexity is better, so merge if merged version has lower perplexity
        if original_perplexity == 0 or merged_perplexity == float('inf'):
            return False
        
        improvement_ratio = original_perplexity / merged_perplexity
        return improvement_ratio > threshold


class ContextualMerger:
    """Contextual token merger using n-gram model."""
    
    def __init__(self, config):
        self.config = config
        self.ngram_model = NGramModel(config.ngram_order)
        
        # Merge statistics
        self.merge_candidates: Dict[Tuple[str, ...], MergeCandidate] = {}
        self.merge_history: List[Tuple[str, ...]] = []
        
        # Common multi-word expressions (seed patterns)
        self.seed_patterns = {
            # Common phrases
            ('at', 'the', 'same', 'time'): 'at_the_same_time',
            ('in', 'order', 'to'): 'in_order_to',
            ('as', 'well', 'as'): 'as_well_as',
            ('on', 'the', 'other', 'hand'): 'on_the_other_hand',
            
            # Technical terms
            ('machine', 'learning'): 'machine_learning',
            ('artificial', 'intelligence'): 'artificial_intelligence',
            ('neural', 'network'): 'neural_network',
            ('deep', 'learning'): 'deep_learning',
            
            # Common verbs + prepositions
            ('look', 'at'): 'look_at',
            ('think', 'about'): 'think_about',
            ('work', 'on'): 'work_on',
            ('focus', 'on'): 'focus_on',
        }
        
        # Initialize with seed patterns
        for tokens, merged in self.seed_patterns.items():
            candidate = MergeCandidate(
                tokens=tokens,
                frequency=10,  # Give seed patterns initial frequency
                contexts=[],
                score=1.0
            )
            self.merge_candidates[tokens] = candidate
    
    def train_on_sequences(self, token_sequences: List[List[str]]) -> None:
        """Train the merger on token sequences."""
        # Train n-gram model
        self.ngram_model.train(token_sequences)
        
        # Find merge candidates
        self._find_merge_candidates(token_sequences)
    
    def _find_merge_candidates(self, token_sequences: List[List[str]]) -> None:
        """Find potential merge candidates from token sequences."""
        # Count n-gram frequencies
        ngram_counts = defaultdict(int)
        ngram_contexts = defaultdict(list)
        
        for sequence in token_sequences:
            for length in range(2, min(len(sequence) + 1, self.config.max_merge_length + 1)):
                for i in range(len(sequence) - length + 1):
                    ngram = tuple(sequence[i:i + length])
                    ngram_counts[ngram] += 1
                    
                    # Store context (surrounding tokens)
                    context_start = max(0, i - 2)
                    context_end = min(len(sequence), i + length + 2)
                    context = ' '.join(sequence[context_start:context_end])
                    ngram_contexts[ngram].append(context)
        
        # Create merge candidates
        for ngram, frequency in ngram_counts.items():
            if frequency >= 3 and len(ngram) >= 2:  # Minimum frequency and length
                score = self._calculate_merge_score(ngram, frequency, ngram_contexts[ngram])
                
                if score >= self.config.merge_threshold:
                    candidate = MergeCandidate(
                        tokens=ngram,
                        frequency=frequency,
                        contexts=ngram_contexts[ngram][:5],  # Keep top 5 contexts
                        score=score
                    )
                    self.merge_candidates[ngram] = candidate
    
    def _calculate_merge_score(self, tokens: Tuple[str, ...], frequency: int, 
                              contexts: List[str]) -> float:
        """Calculate merge score for token sequence."""
        # Base score from frequency
        base_score = min(1.0, frequency / 10.0)
        
        # Bonus for consistent contexts
        context_diversity = len(set(contexts)) / max(1, len(contexts))
        context_bonus = 1.0 - context_diversity  # Lower diversity = higher bonus
        
        # Penalty for very long sequences
        length_penalty = 1.0 / len(tokens) if len(tokens) > 2 else 1.0
        
        # N-gram model score
        ngram_score = 1.0
        if self.config.enable_contextual_merging:
            try:
                # Check if merging improves perplexity
                if self.ngram_model.should_merge(list(tokens), self.config.merge_threshold):
                    ngram_score = 1.2  # Bonus for good n-gram fit
                else:
                    ngram_score = 0.8  # Penalty for poor n-gram fit
            except:
                ngram_score = 1.0  # Neutral if calculation fails
        
        final_score = base_score * (1 + context_bonus * 0.3) * length_penalty * ngram_score
        return min(1.0, final_score)
    
    def merge_tokens(self, tokens: List[str]) -> List[str]:
        """Apply contextual merging to a token sequence."""
        if not self.config.enable_contextual_merging or len(tokens) < 2:
            return tokens
        
        merged_tokens = tokens.copy()
        changes_made = True
        
        # Apply merging iteratively
        while changes_made:
            changes_made = False
            new_tokens = []
            i = 0
            
            while i < len(merged_tokens):
                best_merge = None
                best_length = 0
                
                # Look for merge candidates starting at position i
                for length in range(min(self.config.max_merge_length, len(merged_tokens) - i), 1, -1):
                    candidate_tokens = tuple(merged_tokens[i:i + length])
                    
                    if candidate_tokens in self.merge_candidates:
                        candidate = self.merge_candidates[candidate_tokens]
                        if candidate.score >= self.config.merge_threshold:
                            best_merge = candidate
                            best_length = length
                            break
                
                if best_merge:
                    # Apply merge
                    new_tokens.append(best_merge.merged_token)
                    i += best_length
                    changes_made = True
                    
                    # Update merge statistics
                    best_merge.frequency += 1
                    self.merge_history.append(best_merge.tokens)
                else:
                    # No merge found, keep original token
                    new_tokens.append(merged_tokens[i])
                    i += 1
            
            merged_tokens = new_tokens
        
        return merged_tokens
    
    def add_merge_rule(self, tokens: List[str], merged_token: str = None) -> None:
        """Manually add a merge rule."""
        if len(tokens) < 2:
            return
        
        tokens_tuple = tuple(tokens)
        if merged_token is None:
            merged_token = '_'.join(tokens)
        
        candidate = MergeCandidate(
            tokens=tokens_tuple,
            frequency=5,  # Give manual rules initial frequency
            contexts=[],
            score=1.0  # Maximum score for manual rules
        )
        candidate.merged_token = merged_token
        self.merge_candidates[tokens_tuple] = candidate
    
    def get_merge_stats(self) -> Dict:
        """Get merging statistics."""
        return {
            'total_candidates': len(self.merge_candidates),
            'total_merges_applied': len(self.merge_history),
            'top_merges': [
                {
                    'tokens': list(candidate.tokens),
                    'merged_token': candidate.merged_token,
                    'frequency': candidate.frequency,
                    'score': candidate.score
                }
                for candidate in sorted(
                    self.merge_candidates.values(),
                    key=lambda x: x.frequency,
                    reverse=True
                )[:10]
            ]
        }
    
    def clear_low_frequency_candidates(self, min_frequency: int = 2) -> None:
        """Remove low-frequency merge candidates."""
        to_remove = [
            tokens for tokens, candidate in self.merge_candidates.items()
            if candidate.frequency < min_frequency and tokens not in self.seed_patterns
        ]
        
        for tokens in to_remove:
            del self.merge_candidates[tokens]
