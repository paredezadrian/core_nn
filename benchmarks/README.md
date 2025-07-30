# CORE-NN Memory System Benchmarks

This directory contains comprehensive benchmarks and tests for the CORE-NN memory system, specifically testing the integration of `#remember()`, `#recall()`, and `#forget()` functionality.

## 📁 Files Overview

### Core Test Files

- **`test_memory.py`** - Comprehensive memory system benchmark with 6 test categories
- **`focused_memory_test.py`** - Focused test following the exact pseudocode pattern
- **`run_memory_benchmark.py`** - Benchmark runner with command-line options

### Test Categories

1. **Basic Command Handling** - Tests `#remember`, `#recall`, `#forget` commands
2. **Dynamic Tokenization** - Tests system command tokenization and processing
3. **Session-Based Recall** - Tests memory persistence within sessions
4. **Memory Integration** - Tests BCM and IGPM integration
5. **Performance Stress** - Tests performance under load
6. **Edge Cases** - Tests error handling and malformed inputs

## 🚀 Quick Start

### Run the Focused Test (Pseudocode Pattern)

```bash
# Run the exact test pattern from requirements
python benchmarks/focused_memory_test.py
```

This test follows the pattern:
```python
model = load_core_nn_model()
model("define core-nn")
model("#remember(core-nn)")
output = model("what is core-nn?")
assert "define core-nn" in output
```

### Run Full Benchmark Suite

```bash
# Run all memory system tests
python benchmarks/test_memory.py

# Or use the runner with options
python benchmarks/run_memory_benchmark.py

# Quick mode (skip stress tests)
python benchmarks/run_memory_benchmark.py --quick

# Verbose output
python benchmarks/run_memory_benchmark.py --verbose
```

## 📊 Test Results

### Expected Outputs

**Successful Test Run:**
```
🧪 Running: Basic Command Handling
✅ Basic Command Handling - 0.123s, 2.45MB

🧪 Running: Dynamic Tokenization  
✅ Dynamic Tokenization - 0.089s, 1.23MB

📊 BENCHMARK SUMMARY
Total Tests: 6
Passed: 6 ✅
Failed: 0 ❌
Success Rate: 100.0%
🎉 ALL TESTS PASSED!
```

**Results Files:**
- `memory_benchmark_results.json` - Detailed benchmark results
- `focused_test_results.json` - Focused test results

## 🧪 Test Scenarios

### 1. Basic Command Handling

Tests the core memory operations:

```python
# Remember information
'#remember("CORE-NN is a Context-Oriented Recurrent Embedding Neural Network")'

# Recall information
'#recall("CORE-NN")'

# Forget information  
'#forget("CORE-NN")'
```

**Verification:**
- ✅ Remember command stores information
- ✅ Recall command retrieves stored information
- ✅ Forget command removes information
- ✅ Recall after forget returns "No memories found"

### 2. Dynamic Tokenization

Tests system command tokenization:

```python
commands = [
    '#remember("test content")',
    '#recall("query")', 
    '#forget("term")',
    'Regular text #remember("mixed") with commands'
]
```

**Verification:**
- ✅ System commands are properly tokenized
- ✅ System tokens are detected in token stream
- ✅ Commands are processed by inference engine
- ✅ Mixed text with commands handled correctly

### 3. Session-Based Recall

Tests memory persistence within sessions:

```python
# Store multiple memories
memories = [
    "Python is a programming language",
    "Machine learning uses neural networks",
    "CORE-NN is designed for edge devices"
]

# Test cross-memory recall
recall_tests = [
    ("Python", "programming language"),
    ("neural", "networks"), 
    ("CORE-NN", "edge devices")
]
```

**Verification:**
- ✅ Multiple memories stored successfully
- ✅ Cross-memory recall works (query one term, find related)
- ✅ Session statistics tracked correctly
- ✅ Memory relationships maintained

### 4. Memory Integration

Tests BCM and IGPM integration:

**Verification:**
- ✅ BCM memory count increases after storage
- ✅ IGPM episodic memories increase after storage
- ✅ EpisodicStore statistics show activity
- ✅ Direct model methods work alongside engine methods
- ✅ Recall consistency between direct and engine methods

### 5. Performance Stress Test

Tests system under load:

```python
# Rapid memory operations
for i in range(50):
    if i % 2 == 0:
        engine.generate(f'#remember("Stress test memory {i}")')
    else:
        engine.generate(f'#recall("Stress test")')
```

**Verification:**
- ✅ System handles rapid operations
- ✅ Performance metrics within acceptable range
- ✅ Memory cleanup works after stress test
- ✅ No memory leaks or crashes

### 6. Edge Cases & Error Handling

Tests robustness:

```python
edge_cases = [
    '#remember("")',           # Empty content
    '#recall()',              # Missing arguments
    '#remember("unclosed',    # Malformed syntax
    '#invalid_command("test")'  # Unknown command
]
```

**Verification:**
- ✅ Empty commands handled gracefully
- ✅ Malformed commands don't crash system
- ✅ Unknown commands ignored or handled
- ✅ Very long content processed correctly

## 🔧 Configuration

### Environment Setup

```bash
# Ensure CORE-NN is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/core-nn"

# Or run from project root
cd /path/to/core-nn
python benchmarks/test_memory.py
```

### Custom Configuration

Modify test parameters in the benchmark files:

```python
# In test_memory.py
num_operations = 50  # Stress test operations
top_k = 5           # Recall limit

# In focused_memory_test.py  
key_terms = [        # Terms to verify in recall
    "Context-Oriented",
    "Neural Network",
    "edge"
]
```

## 📈 Performance Expectations

### Typical Performance Metrics

- **Command Processing**: < 0.1s per command
- **Memory Storage**: < 0.05s per remember operation
- **Memory Recall**: < 0.1s per recall operation
- **Memory Usage**: < 5MB per test suite
- **Stress Test**: > 100 operations/second

### Success Criteria

- ✅ All basic commands work correctly
- ✅ System commands properly tokenized
- ✅ Memory persists within sessions
- ✅ BCM and IGPM integration functional
- ✅ Performance within acceptable limits
- ✅ Error handling robust

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure CORE-NN modules are importable
python -c "import core_nn; print('✅ Import successful')"
```

**Memory Errors:**
```bash
# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1024/1024/1024:.1f}GB')"
```

**Configuration Issues:**
```bash
# Verify configuration loading
python -c "from core_nn import ConfigManager; print('✅ Config loading works')"
```

### Debug Mode

Enable verbose logging in tests:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Adding New Tests

### Test Template

```python
def test_new_functionality(self) -> Dict[str, Any]:
    """Test new memory functionality."""
    results = {}
    
    # Test setup
    test_input = '#remember("test data")'
    
    # Execute test
    result = self.engine.generate(input_text=test_input)
    
    # Verify results
    results['success'] = "expected_output" in result.generated_text
    results['response'] = result.generated_text
    
    return results
```

### Integration

Add to test suite in `test_memory.py`:

```python
test_functions = [
    # ... existing tests ...
    (self.test_new_functionality, "New Functionality Test")
]
```

## 📚 References

- [CORE-NN Architecture](../docs/architecture.md)
- [Memory Systems Guide](../docs/memory_systems.md)
- [Tokenizer Documentation](../docs/tokenizer_guide.md)
- [API Reference](../docs/api_reference.md)
