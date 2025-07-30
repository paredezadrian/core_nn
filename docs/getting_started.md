# Getting Started with CORE-NN

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/core-nn.git
cd core-nn
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install CORE-NN:**
```bash
pip install -e .
```

### Initialize a Project

```bash
# Initialize with default configuration
core-nn init

# Initialize with edge device configuration
core-nn init --config-template edge_device

# Initialize in specific directory
core-nn init --output-dir my_project
```

### Start Interactive Chat

```bash
# Start chat with default configuration
core-nn chat

# Start chat with custom configuration
core-nn chat --config configs/edge_device.yaml

# Start chat with specific session name
core-nn chat --session-name my_session
```

## Basic Usage

### Python API

```python
from core_nn import CoreNNModel, ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config('configs/default.yaml')

# Create model
model = CoreNNModel(config)

# Start session
model.start_session()

# Remember information
model.remember("The capital of France is Paris")

# Recall information
memories = model.recall("capital of France")
print(memories)

# Generate text
import torch
input_ids = torch.tensor([[1, 2, 3, 4]])  # Your tokenized input
result = model.generate(input_ids, max_new_tokens=50)
print(result['generated_text'])
```

### Command Line Interface

```bash
# Get system information
core-nn info

# Validate configuration
core-nn validate --config-file configs/my_config.yaml

# Optimize configuration for deployment
core-nn optimize-config --input-file configs/default.yaml \
                       --output-file configs/optimized.yaml \
                       --deployment edge
```

## Configuration

### Configuration Files

CORE-NN uses YAML configuration files to manage model parameters:

```yaml
# configs/my_config.yaml
model:
  name: "my-core-nn"
  version: "1.0.0"

bcm:
  memory_size: 512
  embedding_dim: 768
  salience_threshold: 0.7

rteu:
  num_layers: 4
  embedding_dim: 768
  hidden_dim: 2048

# ... other components
```

### Environment Variables

Override configuration with environment variables:

```bash
export CORE_NN_BCM_MEMORY_SIZE=256
export CORE_NN_DEVICE_PREFERRED=cpu
export CORE_NN_EXECUTION_ENGINE_MEMORY_BUDGET_GB=8
```

### Configuration Templates

Available templates:
- `default`: Balanced configuration for most use cases
- `edge_device`: Optimized for laptops and edge devices
- `minimal`: Minimal resource usage for testing

## Memory Operations

### Remember Information

```python
# Explicit memory storage
model.remember("Important fact to remember")

# With context
model.remember("Meeting at 3 PM", context={"type": "appointment"})
```

### Recall Information

```python
# Search memories
memories = model.recall("meeting", top_k=5)

# Access episodic memories
for memory in memories['episodic_memories']:
    print(f"Instruction: {memory.instruction}")
    print(f"Timestamp: {memory.timestamp}")
```

### Forget Information

```python
# Remove memories related to a topic
result = model.forget("old project")
print(f"Removed {result['igpm_memories_removed']} memories")
```

## Text Generation

### Basic Generation

```python
# Simple generation
result = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.7
)
```

### Advanced Generation

```python
from core_nn.inference import GenerationConfig

# Custom generation configuration
gen_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1
)

result = model.generate(
    input_ids=input_ids,
    generation_config=gen_config,
    instruction="Write a creative story"
)
```

## Session Management

### Creating Sessions

```python
from core_nn.inference import SessionManager

session_manager = SessionManager(config.session)
session = session_manager.create_session("My Session")
```

### Managing Interactions

```python
# Add interaction to session
session.add_interaction(
    user_input="Hello, how are you?",
    model_response="I'm doing well, thank you!",
    metadata={"response_time": 0.5}
)

# Get recent interactions
recent = session.get_recent_interactions(count=5)

# Search interactions
matches = session.search_interactions("hello")
```

### Session Persistence

```python
# Save session
session_manager.save_session(session)

# Load session
loaded_session = session_manager.load_session(session_id)

# List all sessions
sessions = session_manager.list_sessions()
```

## Performance Optimization

### Memory Optimization

```python
# Optimize memory usage
optimization_result = model.optimize_memory()
print(optimization_result)

# Get memory statistics
stats = model.get_memory_stats()
print(f"BCM memories: {stats['bcm_stats']['num_memories']}")
```

### Inference Optimization

```python
from core_nn.inference import InferenceEngine

# Create optimized inference engine
inference_engine = InferenceEngine(model, config.inference)
inference_engine.optimize_for_inference()

# Benchmark performance
benchmark_results = inference_engine.benchmark(num_runs=10)
print(f"Average tokens/sec: {benchmark_results['average_tokens_per_second']}")
```

## Monitoring and Debugging

### System Status

```python
# Get comprehensive system status
status = model.execution_engine.get_system_status()
print(f"Memory usage: {status['system']['memory_percent']}%")
print(f"Active modules: {status['modules']['active_modules']}")
```

### Logging

```python
from core_nn.utils import setup_logging

# Setup detailed logging
setup_logging(level="DEBUG", log_file="core_nn.log")
```

### Profiling

```python
from core_nn.utils import ProfilerContext

# Profile operations
with ProfilerContext(name="Generation"):
    result = model.generate(input_ids, max_new_tokens=50)
```

## Deployment

### Edge Device Deployment

```python
# Load edge-optimized configuration
config = config_manager.load_config('configs/edge_device.yaml')

# Create deployment-optimized configuration
edge_config = config_manager.create_deployment_config(config, 'edge')

# Initialize model with optimized config
model = CoreNNModel(edge_config)
```

### Server Deployment

```python
# Server optimization
server_config = config_manager.create_deployment_config(config, 'server')
model = CoreNNModel(server_config)

# Enable compilation for better performance
if hasattr(torch, 'compile'):
    model = torch.compile(model)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `memory_budget_gb` in execution engine config
   - Use edge device configuration template
   - Enable memory optimization

2. **Slow Performance**
   - Check CPU usage and reduce `cpu_threads` if needed
   - Enable model compilation
   - Use smaller model configurations

3. **Configuration Errors**
   - Validate configuration with `core-nn validate`
   - Check environment variable overrides
   - Ensure all required sections are present

### Getting Help

- Check the [Architecture Guide](architecture.md) for detailed component information
- Review [API Documentation](api.md) for complete API reference
- Run tests with `pytest tests/` to verify installation
- Use `core-nn info` to check system compatibility

## Next Steps

1. **Explore Examples**: Check out `examples/` directory for more usage patterns
2. **Read Architecture Guide**: Understand the underlying components and design
3. **Run Tests**: Execute the test suite to verify everything works
4. **Customize Configuration**: Create your own configuration for specific use cases
5. **Contribute**: See `CONTRIBUTING.md` for development guidelines
