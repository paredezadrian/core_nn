"""
Utility modules for CORE-NN.

Provides logging, device management, profiling, and other utility functions.
"""

from .logging import setup_logging, get_logger
from .device import get_optimal_device, get_device_info
from .profiling import profile_memory, profile_compute, ProfilerContext

__all__ = [
    "setup_logging",
    "get_logger", 
    "get_optimal_device",
    "get_device_info",
    "profile_memory",
    "profile_compute",
    "ProfilerContext",
]
