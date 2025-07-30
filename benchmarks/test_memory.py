#!/usr/bin/env python3
"""
Memory System Benchmark and Test Loop for CORE-NN.

This benchmark verifies that all memory components are properly wired:
- Command handling (#remember, #recall, #forget)
- Dynamic tokenization with system commands
- Session-based recall and memory persistence
- Integration between tokenizer, engine, and memory systems
"""

import sys
import time
import torch
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add the core_nn directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_nn import CoreNNModel, ConfigManager
from core_nn.inference.engine import InferenceEngine
from core_nn.inference.session import SessionManager


@dataclass
class BenchmarkResult:
    """Result from a benchmark test."""
    test_name: str
    success: bool
    execution_time: float
    memory_usage_mb: float
    details: Dict[str, Any]
    error_message: str = ""


class MemoryBenchmark:
    """Comprehensive memory system benchmark."""
    
    def __init__(self):
        """Initialize the benchmark."""
        self.results: List[BenchmarkResult] = []
        self.model = None
        self.engine = None
        self.session_manager = None
        
    def setup(self):
        """Setup the benchmark environment."""
        print("🔧 Setting up CORE-NN Memory Benchmark...")
        
        # Initialize configuration
        config_manager = ConfigManager()
        # Try to load default config file, fallback to default_config attribute
        try:
            default_config_path = Path("configs/default.yaml")
            if default_config_path.exists():
                config = config_manager.load_config(default_config_path)
            else:
                config = config_manager.default_config
        except Exception as e:
            print(f"Warning: Could not load config file, using default: {e}")
            config = config_manager.default_config
        
        # Create model
        print("📦 Creating CORE-NN model...")
        self.model = CoreNNModel(config)
        
        # Create inference engine
        print("🚀 Creating inference engine...")
        self.engine = InferenceEngine(self.model, config.inference, self.model.tokenizer)
        
        # Create session manager
        print("📋 Creating session manager...")
        self.session_manager = SessionManager(config.session)
        
        print("✅ Setup complete!\n")
    
    def run_benchmark(self, test_func, test_name: str) -> BenchmarkResult:
        """Run a single benchmark test."""
        print(f"🧪 Running: {test_name}")
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            details = test_func()
            success = True
            error_message = ""
        except Exception as e:
            details = {"error": str(e)}
            success = False
            error_message = str(e)
            print(f"❌ Test failed: {e}")
        
        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - initial_memory
        
        result = BenchmarkResult(
            test_name=test_name,
            success=success,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            details=details,
            error_message=error_message
        )
        
        self.results.append(result)
        
        if success:
            print(f"✅ {test_name} - {execution_time:.3f}s, {memory_usage:.2f}MB")
        else:
            print(f"❌ {test_name} - FAILED: {error_message}")
        
        return result
    
    def test_basic_command_handling(self) -> Dict[str, Any]:
        """Test basic memory command handling."""
        results = {}
        
        # Test #remember command
        remember_result = self.engine.generate(
            input_text='#remember("CORE-NN is a Context-Oriented Recurrent Embedding Neural Network")'
        )
        results['remember_success'] = "Remembered:" in remember_result.generated_text
        results['remember_text'] = remember_result.generated_text
        
        # Test #recall command
        recall_result = self.engine.generate(input_text='#recall("CORE-NN")')
        results['recall_success'] = "Context-Oriented" in recall_result.generated_text or "CORE-NN" in recall_result.generated_text
        results['recall_text'] = recall_result.generated_text
        
        # Test #forget command
        forget_result = self.engine.generate(input_text='#forget("CORE-NN")')
        results['forget_success'] = "Forgot" in forget_result.generated_text
        results['forget_text'] = forget_result.generated_text
        
        # Verify forget worked
        recall_after_forget = self.engine.generate(input_text='#recall("CORE-NN")')
        results['forget_verified'] = "No memories found" in recall_after_forget.generated_text
        results['recall_after_forget_text'] = recall_after_forget.generated_text
        
        return results
    
    def test_dynamic_tokenization(self) -> Dict[str, Any]:
        """Test dynamic tokenization with system commands."""
        results = {}
        tokenizer = self.model.tokenizer
        
        # Test system command tokenization
        test_commands = [
            '#remember("test content")',
            '#recall("query")',
            '#forget("term")',
            '#remember("simple command without mixed text")',
        ]
        
        for i, command in enumerate(test_commands):
            # Tokenize
            tokens = tokenizer.tokenize(command, add_special_tokens=True)
            results[f'command_{i}_tokens'] = len(tokens)
            
            # Check for system tokens
            system_tokens = []
            for token_id in tokens:
                if hasattr(tokenizer, 'vocabulary'):
                    token = tokenizer.vocabulary.get_token(token_id)
                    if token and token.startswith('#'):
                        system_tokens.append(token)
            
            results[f'command_{i}_system_tokens'] = system_tokens
            
            # Test detokenization
            detokenized = tokenizer.detokenize(tokens, skip_special_tokens=False)
            results[f'command_{i}_detokenized'] = detokenized
            
            # Test engine processing
            engine_result = self.engine.generate(input_text=command)
            results[f'command_{i}_processed'] = engine_result.generated_text
        
        return results
    
    def test_session_based_recall(self) -> Dict[str, Any]:
        """Test session-based memory recall and persistence."""
        results = {}
        
        # Create a test session
        session = self.session_manager.create_session("memory_test_session")
        results['session_created'] = session is not None
        
        # Store multiple memories in sequence
        memories_to_store = [
            "Python is a programming language",
            "Machine learning uses neural networks", 
            "CORE-NN is designed for edge devices",
            "Transformers use attention mechanisms",
            "Memory systems enable learning"
        ]
        
        for i, memory in enumerate(memories_to_store):
            result = self.engine.generate(input_text=f'#remember("{memory}")')
            results[f'store_memory_{i}'] = "Remembered:" in result.generated_text
            
            # Add to session
            session.add_interaction(f'#remember("{memory}")', result.generated_text)
        
        # Test recall of different terms
        recall_tests = [
            ("Python", "programming language"),
            ("neural", "networks"),
            ("CORE-NN", "edge devices"),
            ("attention", "mechanisms"),
            ("memory", "learning")
        ]
        
        for i, (query, expected_content) in enumerate(recall_tests):
            recall_result = self.engine.generate(input_text=f'#recall("{query}")')
            results[f'recall_test_{i}_query'] = query
            results[f'recall_test_{i}_found'] = expected_content.lower() in recall_result.generated_text.lower()
            results[f'recall_test_{i}_text'] = recall_result.generated_text
            
            # Add to session
            session.add_interaction(f'#recall("{query}")', recall_result.generated_text)
        
        # Test session statistics
        session_summary = session.get_session_summary()
        results['session_interactions'] = session_summary.get('interaction_count', 0)
        results['session_stats'] = session_summary
        
        return results
    
    def test_memory_integration(self) -> Dict[str, Any]:
        """Test integration between different memory components."""
        results = {}
        
        # Test BCM and IGPM integration
        bcm_stats_before = self.model.bcm.get_memory_stats()
        igpm_stats_before = self.model.igpm.get_plasticity_stats()
        
        # Store memory using engine
        self.engine.generate(input_text='#remember("Integration test: BCM and IGPM working together")')
        
        # Check that both systems were updated
        bcm_stats_after = self.model.bcm.get_memory_stats()
        igpm_stats_after = self.model.igpm.get_plasticity_stats()
        
        results['bcm_memories_increased'] = bcm_stats_after['num_memories'] > bcm_stats_before['num_memories']
        results['igpm_memories_increased'] = igpm_stats_after['episodic_memories'] > igpm_stats_before['episodic_memories']
        
        # Test episodic store statistics
        episodic_stats = self.engine.episodic_store.get_stats()
        results['episodic_store_stats'] = episodic_stats
        results['episodic_store_active'] = episodic_stats['total_entries'] > 0
        
        # Test direct model methods vs engine methods
        direct_result = self.model.remember("Direct model memory test")
        engine_result = self.engine.generate(input_text='#remember("Engine memory test")')
        
        results['direct_model_works'] = direct_result is not None
        results['engine_method_works'] = "Remembered:" in engine_result.generated_text
        
        # Test recall consistency
        direct_recall = self.model.recall("Direct model")
        engine_recall = self.engine.generate(input_text='#recall("Engine")')
        
        results['direct_recall_found'] = len(direct_recall.get('episodic_memories', [])) > 0
        results['engine_recall_found'] = "memories for" in engine_recall.generated_text
        
        return results
    
    def test_performance_stress(self) -> Dict[str, Any]:
        """Test performance under stress conditions."""
        results = {}
        
        # Stress test: rapid memory operations
        num_operations = 50
        start_time = time.time()
        
        for i in range(num_operations):
            # Alternate between remember and recall
            if i % 2 == 0:
                self.engine.generate(input_text=f'#remember("Stress test memory {i}")')
            else:
                self.engine.generate(input_text=f'#recall("Stress test")')
        
        stress_time = time.time() - start_time
        results['stress_operations'] = num_operations
        results['stress_time'] = stress_time
        results['operations_per_second'] = num_operations / stress_time
        
        # Memory usage after stress test
        final_stats = self.engine.episodic_store.get_stats()
        results['final_memory_stats'] = final_stats
        
        # Test memory cleanup
        cleanup_result = self.engine.generate(input_text='#forget("Stress test")')
        results['cleanup_performed'] = "Forgot" in cleanup_result.generated_text
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling."""
        results = {}
        
        # Test empty commands
        empty_tests = [
            '#remember("")',
            '#recall("")',
            '#forget("")',
            '#remember()',
            '#recall()',
            '#forget()'
        ]
        
        for i, test in enumerate(empty_tests):
            try:
                result = self.engine.generate(input_text=test)
                results[f'empty_test_{i}_handled'] = True
                results[f'empty_test_{i}_response'] = result.generated_text
            except Exception as e:
                results[f'empty_test_{i}_handled'] = False
                results[f'empty_test_{i}_error'] = str(e)
        
        # Test malformed commands
        malformed_tests = [
            '#remember("unclosed quote',
            '#recall(malformed)',
            '#forget("too" "many" "quotes")',
            '#invalid_command("test")'
        ]
        
        for i, test in enumerate(malformed_tests):
            try:
                result = self.engine.generate(input_text=test)
                results[f'malformed_test_{i}_handled'] = True
                results[f'malformed_test_{i}_response'] = result.generated_text
            except Exception as e:
                results[f'malformed_test_{i}_handled'] = False
                results[f'malformed_test_{i}_error'] = str(e)
        
        # Test very long content
        long_content = "Very long memory content " * 100
        try:
            result = self.engine.generate(input_text=f'#remember("{long_content}")')
            results['long_content_handled'] = True
            results['long_content_response'] = result.generated_text[:100] + "..."
        except Exception as e:
            results['long_content_handled'] = False
            results['long_content_error'] = str(e)
        
        return results
    
    def run_all_tests(self):
        """Run all benchmark tests."""
        print("🚀 Starting CORE-NN Memory System Benchmark")
        print("=" * 60)
        
        # Setup
        self.setup()
        
        # Run all tests
        test_functions = [
            (self.test_basic_command_handling, "Basic Command Handling"),
            (self.test_dynamic_tokenization, "Dynamic Tokenization"),
            (self.test_session_based_recall, "Session-Based Recall"),
            (self.test_memory_integration, "Memory Integration"),
            (self.test_performance_stress, "Performance Stress Test"),
            (self.test_edge_cases, "Edge Cases & Error Handling")
        ]
        
        for test_func, test_name in test_functions:
            self.run_benchmark(test_func, test_name)
            print()  # Add spacing between tests
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate benchmark summary."""
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_time = sum(r.execution_time for r in self.results)
        total_memory = sum(r.memory_usage_mb for r in self.results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Execution Time: {total_time:.3f}s")
        print(f"Total Memory Usage: {total_memory:.2f}MB")
        print(f"Average Time per Test: {total_time/total_tests:.3f}s")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.test_name}: {result.error_message}")
        
        # Save detailed results
        self.save_results()
        
        print(f"\n{'🎉 ALL TESTS PASSED!' if failed_tests == 0 else '⚠️  SOME TESTS FAILED'}")
        print("=" * 60)
    
    def save_results(self):
        """Save detailed benchmark results to file."""
        results_file = Path("benchmarks/memory_benchmark_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'test_name': result.test_name,
                'success': result.success,
                'execution_time': result.execution_time,
                'memory_usage_mb': result.memory_usage_mb,
                'details': result.details,
                'error_message': result.error_message
            })
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'total_tests': len(self.results),
                'passed_tests': sum(1 for r in self.results if r.success),
                'results': serializable_results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024


if __name__ == "__main__":
    benchmark = MemoryBenchmark()
    benchmark.run_all_tests()
