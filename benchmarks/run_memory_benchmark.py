#!/usr/bin/env python3
"""
Memory Benchmark Runner

Simple script to run the memory system benchmark with different configurations.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.test_memory import MemoryBenchmark


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run CORE-NN Memory System Benchmark")
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run quick benchmark (skip stress tests)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--save-results", 
        action="store_true", 
        default=True,
        help="Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    print("🚀 CORE-NN Memory System Benchmark Runner")
    print("=" * 50)
    
    if args.quick:
        print("⚡ Running in QUICK mode (skipping stress tests)")
    
    if args.verbose:
        print("📝 Verbose output enabled")
    
    # Create and run benchmark
    benchmark = MemoryBenchmark()
    
    if args.quick:
        # Run only essential tests
        benchmark.setup()
        
        essential_tests = [
            (benchmark.test_basic_command_handling, "Basic Command Handling"),
            (benchmark.test_dynamic_tokenization, "Dynamic Tokenization"),
            (benchmark.test_memory_integration, "Memory Integration"),
        ]
        
        for test_func, test_name in essential_tests:
            benchmark.run_benchmark(test_func, test_name)
            print()
        
        benchmark.generate_summary()
    else:
        # Run full benchmark suite
        benchmark.run_all_tests()


if __name__ == "__main__":
    main()
