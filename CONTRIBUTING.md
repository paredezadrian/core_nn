# Contributing to CORE-NN

Thank you for your interest in contributing to CORE-NN! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/core_nn.git
   cd core_nn
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -e .
   pip install pytest pytest-cov black flake8
   ```

5. Run tests to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Feature additions**: Add new functionality
- **Documentation improvements**: Enhance docs, examples, or comments
- **Performance optimizations**: Improve speed or memory usage
- **Research contributions**: Implement new architectural improvements
- **Testing**: Add or improve test coverage

### Priority Areas

Based on our research roadmap, these areas are particularly valuable:

1. **IGPM Plasticity Enhancement** - Currently showing minimal plasticity (0.0000 change)
2. **BCM Salience Optimization** - Improve memory retention mechanisms
3. **Real-world Task Evaluation** - Test on actual language modeling benchmarks
4. **Edge Device Optimization** - Improve performance on resource-constrained devices

## Code Standards

### Python Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black .`
- Use type hints where appropriate
- Maximum line length: 100 characters

### Code Quality

- Write clear, self-documenting code
- Add docstrings to all public functions and classes
- Use meaningful variable and function names
- Keep functions focused and small

### Example Code Style

```python
def process_temporal_sequence(
    embeddings: torch.Tensor,
    temporal_scales: List[int],
    device: torch.device = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Process embeddings through multi-timescale temporal layers.
    
    Args:
        embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
        temporal_scales: List of temporal scales to use
        device: Target device for computation
        
    Returns:
        Tuple of (processed_embeddings, processing_info)
    """
    if device is None:
        device = embeddings.device
    
    # Implementation here...
    return processed_embeddings, info_dict
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core_components.py -v

# Run with coverage
pytest tests/ --cov=core_nn --cov-report=html
```

### Writing Tests

- Write tests for all new functionality
- Aim for >90% test coverage
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common test setup

### Test Structure

```python
import pytest
import torch
from core_nn.components.bcm import BiologicalCoreMemory
from core_nn.config.schema import BCMConfig

class TestBiologicalCoreMemory:
    """Test suite for BCM component."""
    
    @pytest.fixture
    def bcm_config(self):
        return BCMConfig(memory_size=64, embedding_dim=128)
    
    @pytest.fixture
    def bcm(self, bcm_config):
        return BiologicalCoreMemory(bcm_config)
    
    def test_memory_retention(self, bcm):
        """Test that BCM retains high-salience memories."""
        # Test implementation here...
        assert condition
```

## Documentation

### Documentation Standards

- Update documentation for any API changes
- Add examples for new features
- Use clear, concise language
- Include code examples where helpful

### Documentation Types

1. **API Documentation** (`docs/api.md`) - Complete API reference
2. **Architecture Documentation** (`docs/architecture.md`) - Component details
3. **Examples** (`examples/`) - Working code examples
4. **Research Documentation** (`docs/future_research.md`) - Research directions

### Docstring Format

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        RuntimeError: When computation fails
        
    Example:
        >>> result = example_function("test", 20)
        >>> print(result)
        True
    """
    # Implementation here...
```

## Submitting Changes

### Pull Request Process

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes**:
   ```bash
   pytest tests/ -v
   python examples/comprehensive_demo.py
   ```

4. **Update documentation** if needed

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

### Commit Message Format

Use conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes

Examples:
```
feat: add adaptive salience thresholds to BCM
fix: resolve tensor shape mismatch in IGPM
docs: update API documentation for new methods
test: add comprehensive RTEU temporal processing tests
```

### Pull Request Guidelines

- **Clear title and description** explaining the changes
- **Reference related issues** if applicable
- **Include test results** showing all tests pass
- **Update documentation** for any API changes
- **Keep changes focused** - one feature/fix per PR
- **Respond to feedback** promptly and professionally

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Error messages** and stack traces
- **Minimal code example** if possible

### Feature Requests

For feature requests, please include:

- **Clear description** of the proposed feature
- **Use case** explaining why it's needed
- **Proposed implementation** if you have ideas
- **Alternatives considered**

## Research Contributions

### Novel Architecture Research

We particularly welcome contributions in these areas:

1. **Biological Inspiration** - New biologically-inspired mechanisms
2. **Memory Systems** - Improvements to memory retention and recall
3. **Temporal Processing** - Enhanced multi-timescale processing
4. **Edge Optimization** - Better performance on resource-constrained devices
5. **Compression Techniques** - Advanced knowledge compression methods

### Research Contribution Process

1. **Discuss your idea** by opening an issue first
2. **Implement a prototype** with proper testing
3. **Document the approach** and results
4. **Compare against baselines** with benchmarks
5. **Submit a detailed PR** with research documentation

## Getting Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Email** - itsparedezadrian@outlook.com for direct contact

## Recognition

Contributors will be recognized in:

- **README.md** - Major contributors listed
- **CHANGELOG.md** - Contributions noted in releases
- **Research papers** - Academic contributions acknowledged

Thank you for contributing to CORE-NN! Your contributions help advance the field of edge-efficient AI architectures.
