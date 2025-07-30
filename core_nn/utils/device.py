"""
Device management utilities for CORE-NN.
"""

import torch
import platform
from typing import Dict, Any, Optional


def get_optimal_device(preferred: str = "auto") -> torch.device:
    """
    Get optimal device for CORE-NN execution.
    
    Args:
        preferred: Preferred device type ("auto", "cpu", "cuda", "mps")
        
    Returns:
        Optimal torch device
    """
    if preferred == "auto":
        # Auto-detect best available device
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    elif preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    
    elif preferred == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS not available, falling back to CPU")
            return torch.device("cpu")
    
    else:
        return torch.device("cpu")


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Args:
        device: Device to get info for (default: current device)
        
    Returns:
        Dictionary with device information
    """
    if device is None:
        device = get_optimal_device()
    
    info = {
        "device_type": device.type,
        "device_index": device.index,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    if device.type == "cuda":
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "current_gpu": torch.cuda.current_device(),
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": torch.cuda.get_device_properties(device).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(device),
                "gpu_memory_cached": torch.cuda.memory_reserved(device),
            })
    
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })
    
    return info
