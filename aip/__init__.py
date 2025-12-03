# aip/__init__.py
from .core import (
    generate_RIS,
    SignalEmbedNet,
    NodeGraphNet,
    coherence_measure,
    model_checksum,
)

__version__ = "1.0.0"
__all__ = [
    "generate_RIS",
    "SignalEmbedNet",
    "NodeGraphNet",
    "coherence_measure",
    "model_checksum",
]
