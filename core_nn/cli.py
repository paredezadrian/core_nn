"""
Command Line Interface for CORE-NN.

Provides CLI commands for initialization, configuration, training, and interactive chat.
"""

import click
import torch
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os

from .config.manager import ConfigManager
from .config.schema import CoreNNConfig
from .model import CoreNNModel
from .inference.engine import InferenceEngine
from .inference.session import SessionManager
from .utils.logging import setup_logging
from .utils.device import get_optimal_device


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """CORE-NN: Context-Oriented Recurrent Embedding Neural Network CLI."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    # Load configuration
    config_manager = ConfigManager()
    if config:
        ctx.obj['config'] = config_manager.load_config(config)
    else:
        # Use default config
        default_config_path = Path("configs/default.yaml")
        if default_config_path.exists():
            ctx.obj['config'] = config_manager.load_config(default_config_path)
        else:
            ctx.obj['config'] = CoreNNConfig()
    
    ctx.obj['config_manager'] = config_manager


@cli.command()
@click.option('--config-template', '-t', default='default', 
              help='Configuration template to use (default, edge_device, minimal)')
@click.option('--output-dir', '-o', default='.', help='Output directory')
@click.option('--force', '-f', is_flag=True, help='Force overwrite existing files')
@click.pass_context
def init(ctx, config_template, output_dir, force):
    """Initialize a new CORE-NN project."""
    output_path = Path(output_dir)
    config_manager = ctx.obj['config_manager']
    
    click.echo(f"Initializing CORE-NN project in {output_path}")
    
    # Create directory structure
    directories = [
        'configs', 'sessions', 'kpacks', 'logs', 'checkpoints'
    ]
    
    for dir_name in directories:
        dir_path = output_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"Created directory: {dir_path}")
    
    # Create configuration file
    config_file = output_path / 'configs' / 'config.yaml'
    if config_file.exists() and not force:
        click.echo(f"Configuration file already exists: {config_file}")
        click.echo("Use --force to overwrite")
        return
    
    try:
        # Load template configuration
        template_path = Path("configs") / f"{config_template}.yaml"
        if template_path.exists():
            config = config_manager.load_config(template_path)
        else:
            click.echo(f"Template {config_template} not found, using default")
            config = CoreNNConfig()
        
        # Save configuration
        config_manager.save_config(config, config_file)
        click.echo(f"Created configuration: {config_file}")
        
        # Create example scripts
        example_script = output_path / 'example_usage.py'
        with open(example_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Example usage of CORE-NN.
\"\"\"

from core_nn import CoreNNModel, ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config('configs/config.yaml')

# Initialize model
model = CoreNNModel(config)

# Start interactive session
model.start_session()

# Example: Remember something
model.remember("The capital of France is Paris")

# Example: Recall information
memories = model.recall("capital of France")
print(f"Recalled memories: {memories}")

# Example: Generate text
import torch
input_ids = torch.tensor([[1, 2, 3, 4]])  # Example token IDs
result = model.generate(input_ids, max_new_tokens=10)
print(f"Generated: {result['generated_tokens']}")
""")
        
        click.echo(f"Created example script: {example_script}")
        click.echo("✅ CORE-NN project initialized successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error initializing project: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-path', '-m', type=click.Path(), help='Path to saved model')
@click.option('--session-name', '-s', default='default', help='Session name')
@click.option('--max-tokens', default=100, help='Maximum tokens per response')
@click.option('--temperature', default=0.7, help='Sampling temperature')
@click.pass_context
def chat(ctx, model_path, session_name, max_tokens, temperature):
    """Start interactive chat session."""
    config = ctx.obj['config']
    
    click.echo("🧠 Starting CORE-NN Interactive Chat")
    click.echo("Type 'help' for commands, 'quit' to exit")
    click.echo("-" * 50)
    
    try:
        # Initialize model
        if model_path and Path(model_path).exists():
            # Load saved model
            model = torch.load(model_path)
            click.echo(f"Loaded model from {model_path}")
        else:
            # Create new model
            model = CoreNNModel(config)
            click.echo("Created new model")
        
        # Initialize session manager
        session_manager = SessionManager(config.session)
        session = session_manager.create_session(session_name)
        
        # Start model session
        model.start_session()
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                
                elif user_input.startswith('/remember '):
                    instruction = user_input[10:]
                    result = model.remember(instruction)
                    click.echo(f"🧠 Remembered: {instruction}")
                    click.echo(f"   Details: {result}")
                    continue
                
                elif user_input.startswith('/recall '):
                    query = user_input[8:]
                    memories = model.recall(query)
                    click.echo(f"🔍 Recalled memories for '{query}':")
                    for i, memory in enumerate(memories.get('episodic_memories', [])[:3]):
                        click.echo(f"   {i+1}. {memory.instruction}")
                    continue
                
                elif user_input.startswith('/forget '):
                    query = user_input[8:]
                    result = model.forget(query)
                    click.echo(f"🗑️  Forgot memories related to '{query}'")
                    click.echo(f"   Removed: {result}")
                    continue
                
                elif user_input.startswith('/stats'):
                    stats = model.get_memory_stats()
                    click.echo("📊 Memory Statistics:")
                    click.echo(f"   BCM memories: {stats['bcm_stats']['num_memories']}")
                    click.echo(f"   IGPM memories: {stats['igpm_stats']['episodic_memories']}")
                    click.echo(f"   System status: {model.execution_engine.get_system_status()}")
                    continue
                
                # Regular chat
                click.echo("🤖 CORE-NN: Processing...")

                # Use the model's integrated tokenizer
                input_tokens = model.tokenizer.tokenize(user_input, add_special_tokens=True)
                input_ids = torch.tensor([input_tokens])

                # Generate response
                result = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )

                # Use the model's integrated detokenizer
                response = model.tokenizer.detokenize(result['generated_tokens'], skip_special_tokens=True)
                
                click.echo(f"🤖 CORE-NN: {response}")
                
                # Save to session
                session.add_interaction(user_input, response)
                
            except KeyboardInterrupt:
                click.echo("\n\n👋 Goodbye!")
                break
            except Exception as e:
                click.echo(f"❌ Error: {e}", err=True)
        
        # End session
        model.end_session()
        session_manager.save_session(session)
        click.echo("Session saved. Goodbye! 👋")
        
    except Exception as e:
        click.echo(f"❌ Error starting chat: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), required=True,
              help='Input configuration file')
@click.option('--output-file', '-o', type=click.Path(), required=True,
              help='Output configuration file')
@click.option('--deployment', '-d', type=click.Choice(['edge', 'server', 'mobile']),
              default='edge', help='Deployment type')
@click.pass_context
def optimize_config(ctx, input_file, output_file, deployment):
    """Optimize configuration for specific deployment."""
    config_manager = ctx.obj['config_manager']
    
    try:
        # Load base configuration
        base_config = config_manager.load_config(input_file)
        
        # Create optimized configuration
        optimized_config = config_manager.create_deployment_config(base_config, deployment)
        
        # Save optimized configuration
        config_manager.save_config(optimized_config, output_file)
        
        click.echo(f"✅ Optimized configuration for {deployment} deployment")
        click.echo(f"   Input: {input_file}")
        click.echo(f"   Output: {output_file}")
        
        # Show summary
        summary = config_manager.get_config_summary(optimized_config)
        click.echo("\n📋 Configuration Summary:")
        for section, values in summary.items():
            click.echo(f"   {section}:")
            for key, value in values.items():
                click.echo(f"     {key}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Error optimizing configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), required=True,
              help='Configuration file to optimize')
@click.option('--cpu-only', is_flag=True, help='Optimize for CPU-only inference')
@click.option('--memory-efficient', is_flag=True, help='Enable memory-efficient optimizations')
@click.option('--output-file', '-o', type=click.Path(), help='Output optimized configuration file')
@click.option('--benchmark', is_flag=True, help='Run performance benchmark after optimization')
@click.pass_context
def optimize_inference(ctx, config_file, cpu_only, memory_efficient, output_file, benchmark):
    """Optimize inference pipeline for CPU-only setup."""
    config_manager = ctx.obj['config_manager']
    
    try:
        # Load current configuration
        current_config = config_manager.load_config(config_file)
        
        click.echo("🔧 Optimizing inference pipeline...")
        
        # Apply CPU-only optimizations
        if cpu_only:
            click.echo("  - Enabling CPU-only optimizations")
            current_config.device.preferred = "cpu"
            current_config.device.mixed_precision = False
            current_config.device.compile_model = False
            current_config.execution_engine.cpu_threads = 8
            current_config.execution_engine.async_execution = False
        
        # Apply memory-efficient optimizations
        if memory_efficient:
            click.echo("  - Enabling memory-efficient optimizations")
            current_config.memory.working_memory_size = 64  # Reduced from 128
            current_config.memory.episodic_memory_size = 256  # Reduced from 512
            current_config.memory.semantic_memory_size = 1024  # Reduced from 2048
            current_config.memory.memory_consolidation_interval = 25  # More frequent
        
        # Optimize inference settings
        click.echo("  - Optimizing inference parameters")
        current_config.inference.max_sequence_length = 20  # Conservative for laptop
        current_config.inference.temperature = 0.8  # Slightly higher for creativity
        current_config.inference.top_k = 30  # Reduced from 40
        current_config.inference.top_p = 0.85  # Reduced from 0.9
        current_config.inference.repetition_penalty = 1.05  # Reduced from 1.1
        current_config.inference.max_new_tokens = 128  # Reduced from 256
        
        # Optimize component settings for speed
        click.echo("  - Optimizing component settings")
        current_config.bcm.memory_size = 128  # Reduced from 256
        current_config.bcm.salience_threshold = 0.9  # Higher threshold
        current_config.rteu.num_layers = 2  # Reduced from 3
        current_config.rteu.routing_iterations = 1  # Reduced from 2
        current_config.igpm.plastic_slots = 16  # Reduced from 32
        current_config.igpm.max_episodic_memories = 250  # Reduced from 500
        
        # Save optimized configuration
        if output_file:
            config_manager.save_config(current_config, output_file)
            click.echo(f"✅ Optimized configuration saved to: {output_file}")
        else:
            # Overwrite original file
            config_manager.save_config(current_config, config_file)
            click.echo(f"✅ Optimized configuration saved to: {config_file}")
        
        # Show optimization summary
        click.echo("\n📋 Optimization Summary:")
        click.echo(f"  Device: {current_config.device.preferred}")
        click.echo(f"  CPU Threads: {current_config.execution_engine.cpu_threads}")
        click.echo(f"  Memory Size: {current_config.memory.working_memory_size}")
        click.echo(f"  Max Sequence Length: {current_config.inference.max_sequence_length}")
        click.echo(f"  BCM Memory Size: {current_config.bcm.memory_size}")
        click.echo(f"  RTEU Layers: {current_config.rteu.num_layers}")
        click.echo(f"  IGPM Slots: {current_config.igpm.plastic_slots}")
        
        # Run benchmark if requested
        if benchmark:
            click.echo("\n🏃 Running performance benchmark...")
            try:
                # Import and run benchmark
                from benchmarks.performance_benchmark import PerformanceBenchmark
                benchmark = PerformanceBenchmark()
                benchmark.benchmark_components()
                benchmark.benchmark_full_model()
                benchmark.save_results()
                benchmark.print_summary()
                click.echo("✅ Benchmark completed successfully!")
            except Exception as e:
                click.echo(f"⚠️ Benchmark failed: {e}")
        
    except Exception as e:
        click.echo(f"❌ Error optimizing inference: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), required=True,
              help='Configuration file to validate')
@click.pass_context
def validate(ctx, config_file):
    """Validate configuration file."""
    config_manager = ctx.obj['config_manager']
    
    try:
        errors = config_manager.validate_config_file(config_file)
        
        if not errors:
            click.echo(f"✅ Configuration file {config_file} is valid")
        else:
            click.echo(f"❌ Configuration file {config_file} has errors:")
            for error in errors:
                click.echo(f"   - {error}")
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Error validating configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Show system and model information."""
    config = ctx.obj['config']
    
    click.echo("🧠 CORE-NN System Information")
    click.echo("=" * 40)
    
    # System info
    device_info = get_optimal_device()
    click.echo(f"Optimal device: {device_info}")
    click.echo(f"PyTorch version: {torch.__version__}")
    
    # Configuration summary
    config_manager = ctx.obj['config_manager']
    summary = config_manager.get_config_summary(config)
    
    click.echo("\n📋 Current Configuration:")
    for section, values in summary.items():
        click.echo(f"  {section}:")
        for key, value in values.items():
            click.echo(f"    {key}: {value}")
    
    # Available configurations
    available_configs = config_manager.get_available_configs()
    click.echo(f"\n📁 Available configurations: {', '.join(available_configs)}")


def print_help():
    """Print chat help."""
    help_text = """
🤖 CORE-NN Chat Commands:

Basic Commands:
  help                    - Show this help
  quit, exit, bye        - Exit chat

Memory Commands:
  /remember <text>       - Explicitly remember information
  /recall <query>        - Recall memories related to query
  /forget <query>        - Forget memories related to query
  /stats                 - Show memory and system statistics

Just type normally to chat with CORE-NN!
"""
    click.echo(help_text)


def simple_tokenize(text: str, max_length: int = 50) -> list:
    """
    Simple tokenization (PLACEHOLDER - replace with proper tokenizer).

    DESIGN LIMITATION: This is a simplified tokenization approach for demonstration
    purposes only. It has several limitations:

    1. Limited to 50 characters maximum input length
    2. Uses ASCII character codes as token IDs (capped at 999)
    3. No vocabulary mapping or subword tokenization
    4. No handling of special tokens or out-of-vocabulary words
    5. Fixed padding to max_length regardless of actual input length

    For production use, replace with a proper tokenizer like:
    - Hugging Face transformers tokenizer
    - SentencePiece tokenizer
    - Custom vocabulary-based tokenizer

    Args:
        text: Input text to tokenize
        max_length: Maximum sequence length (default: 50)

    Returns:
        List of token IDs (padded to max_length)
    """
    # Convert characters to token IDs (simplified)
    tokens = [min(ord(c), 999) for c in text[:max_length]]
    return tokens + [0] * (max_length - len(tokens))  # Pad


def simple_detokenize(tokens: list) -> str:
    """
    Simple detokenization (PLACEHOLDER - replace with proper tokenizer).

    DESIGN LIMITATION: This is a simplified detokenization approach that:

    1. Only handles printable ASCII characters (32-126)
    2. Filters out padding tokens (0) and invalid tokens
    3. No proper vocabulary mapping or subword reconstruction
    4. May produce garbled output for non-ASCII inputs

    For production use, replace with the corresponding detokenizer for your
    chosen tokenization approach.

    Args:
        tokens: List of token IDs to detokenize

    Returns:
        Reconstructed text string
    """
    # Convert token IDs back to characters (simplified)
    chars = [chr(min(max(token, 32), 126)) for token in tokens if token > 0]
    return ''.join(chars)


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
