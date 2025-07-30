# CORE-NN: Context-Oriented Recurrent Embedding Neural Network

🌐 **Overview**

CORE-NN is a novel AI architecture designed for edge devices that replaces traditional transformer-based LLMs with efficient, memory-conscious components. **All tests passing (66/66)** with validated performance metrics.

**Key Innovations:**
- 🧠 **Salience-Based Memory**: BCM retains only important information vs. full attention
- ⏰ **Multi-Timescale Processing**: RTEU processes temporal patterns at different scales
- 🧩 **Instruction-Guided Plasticity**: IGPM adapts without global weight updates
- 📦 **Dynamic Knowledge Compression**: MLCS achieves 125x compression ratios
- 🚀 **Edge-Efficient Execution**: Asynchronous modular processing

**Validated Performance:**
- **Memory Usage**: ~42MB total (vs GBs for transformers)
- **Throughput**: 36-75 tokens/sec depending on configuration
- **Component Speed**: BCM (0.15ms), RTEU (2.50ms), IGPM (1.45ms)
- **Compression**: Up to 125x with knowledge packs (.kpack files)

## 🧠 Architecture Components

### 1. Biological Core Memory (BCM)
- **Inspiration**: Hippocampus + working memory
- **Structure**: Fixed-size temporal memory block (256–1024 embeddings)
- **Memory Rule**: Slides over input but selectively retains contextual events above salience threshold

### 2. Recursive Temporal Embedding Unit (RTEU)
- **Inspiration**: Combines RNNs, capsule networks, and latent diffusion
- **Purpose**: Replaces self-attention with temporal routing-by-agreement
- **Operation**: Multi-timescale embeddings with fast/slow capsules

### 3. Instruction-Guided Plasticity Module (IGPM)
- **Purpose**: Meta-learning from user input without global weight updates
- **Features**: "Plastic slots" for pattern modifications tied to instructions
- **Mechanism**: Fast weights and sparse updates triggered by commands

### 4. Multi-Level Compression Synthesizer (MLCS)
- **Purpose**: Model bootstrapping and long-range planning
- **Features**: Compresses knowledge into lightweight latent codes (.kpack files)
- **Benefits**: Dynamic loading/unloading of knowledge modules

### 5. Edge-Efficient Modular Execution Engine
- **Operation**: Asynchronous blocks, not lockstep like transformers
- **Features**: Dynamic component offloading and compression
- **API**: `remember()`, `recall()`, `forget()` commands

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize CORE-NN
python -m core_nn.cli init --config configs/default.yaml

# Run interactive session
python -m core_nn.cli chat

# Run with custom config
python -m core_nn.cli chat --config configs/edge_device.yaml
```

## 📦 Installation

```bash
git clone https://github.com/paredezadrian/core_nn.git
cd core_nn
pip install -e .
```

## 🔧 Quick API Reference

### Python API

```python
from core_nn import CoreNNModel, ConfigManager

# Initialize
config = ConfigManager().load_config('configs/default.yaml')
model = CoreNNModel(config)

# Memory operations
model.remember("Important information")
memories = model.recall("information")
model.forget("outdated data")

# Text generation
import torch
input_ids = torch.tensor([[1, 2, 3, 4]])
result = model.generate(input_ids, max_new_tokens=50)
```

### CLI Commands

```bash
# Initialize project
core-nn init --config-template edge_device

# Interactive chat
core-nn chat --session-name my_session

# Validate configuration
core-nn validate --config-file configs/my_config.yaml
```

## 🏗️ Project Structure

```
core-nn/
├── core_nn/                 # Main package
│   ├── components/          # Architecture components
│   ├── config/             # Configuration management
│   ├── inference/          # Inference engine
│   ├── memory/             # Memory systems
│   └── utils/              # Utilities
├── configs/                # Configuration files
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Usage examples
└── benchmarks/             # Performance benchmarks
```

## 📊 Performance Specs

| Component | CPU Usage | RAM | Disk Footprint |
|-----------|-----------|-----|----------------|
| BCM (512 slots) | Low | ~1 GB | None |
| RTEU (4 layers) | Medium | ~5 GB | None |
| IGPM | Low | ~2 GB | Optional |
| MLCS (10 .kpacks) | Variable | ~2-3 GB | ~50 MB/kpack |
| **Total** | **Moderate** | **~10-12 GB** | **<1 GB** |

## 🧪 Testing & Validation

```bash
# Run all tests (66 tests, 100% pass rate)
pytest tests/ -v

# Run specific component tests
pytest tests/test_core_components.py::TestBiologicalCoreMemory -v
pytest tests/test_core_components.py::TestRecursiveTemporalEmbeddingUnit -v
pytest tests/test_core_components.py::TestInstructionGuidedPlasticityModule -v

# Run architecture validation tests
python test_architecture_focused.py

# Run performance benchmarks
python benchmarks/performance_benchmark.py
python benchmarks/run_memory_benchmark.py

# Validate configuration
python -m core_nn.cli validate --config-file configs/default.yaml
```

### Performance Results

- **Component Performance**: BCM (0.15ms), RTEU (2.50ms), IGPM (1.45ms)
- **Full Model Throughput**: 36-75 tokens/sec (config dependent)
- **Memory Efficiency**: ~42MB total memory usage
- **Compression Ratio**: Up to 125x compression with MLCS
- **Test Coverage**: 100% pass rate across all components

## 📚 Documentation

- [🏗️ Architecture Guide](docs/architecture.md) - Detailed component architecture
- [📖 API Reference](docs/api.md) - Complete API documentation
- [🚀 Getting Started](docs/getting_started.md) - Quick start guide
- [⚙️ Configuration Guide](docs/configuration.md) - Configuration options
- [📦 KPack Usage Guide](docs/kpack_usage_guide.md) - Knowledge compression
- [🔤 Tokenizer Guide](docs/tokenizer_guide.md) - Tokenization system
- [🔬 Areas for Further Exploration](docs/future_research.md) - Research directions
- [⚠️ Known Limitations](docs/known_limitations.md) - Current limitations

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 👨‍💻 Author

**Adrian Paredez** ([@paredezadrian](https://github.com/paredezadrian))
- Email: itsparedezadrian@outlook.com
- Repository: https://github.com/paredezadrian/core_nn.git

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 📚 Citation

If you use CORE-NN in your research, please cite:

```bibtex
@software{paredez2024corenn,
  title={CORE-NN: Context-Oriented Recurrent Embedding Neural Network},
  author={Paredez, Adrian},
  year={2024},
  version={0.0.0-beta},
  url={https://github.com/paredezadrian/core_nn.git}
}
```
