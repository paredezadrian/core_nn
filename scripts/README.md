# Development Scripts

This directory contains development and testing scripts used during CORE-NN development.

## Scripts Overview

### Architecture Testing Scripts
- **`test_architecture_focused.py`** - Focused architecture component testing
- **`test_novel_architecture.py`** - Novel architecture validation tests
- **`test_igpm_improvements.py`** - IGPM plasticity improvement validation
- **`test_memory_commands.py`** - Memory system command testing

## Usage

These scripts are primarily for development and debugging purposes. They were used during the development process to validate specific components and improvements.

### Running Scripts

```bash
# From the project root directory
python scripts/test_architecture_focused.py
python scripts/test_igpm_improvements.py
python scripts/test_memory_commands.py
python scripts/test_novel_architecture.py
```

## Note

For production testing, use the organized test suite in the `tests/` directory and the evaluation framework in `evaluation/`.

These scripts represent the development journey and are kept for reference and debugging purposes.
