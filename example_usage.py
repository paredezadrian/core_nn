#!/usr/bin/env python3
"""
Example usage of CORE-NN.
"""

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
