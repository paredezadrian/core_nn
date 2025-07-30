"""
Inference module for CORE-NN.

Provides inference engine, session management, and runtime execution capabilities.
"""

from .engine import InferenceEngine
from .session import SessionManager, Session

__all__ = [
    "InferenceEngine",
    "SessionManager", 
    "Session",
]
