"""
MIRAS research code package.

This package exposes:
- Llama-like macro-architecture (`ModularLlama`, `LlamaMIRASLayer`)
- A placeholder MONETA block (`MonetaBlock`) and helpers to build example models
"""

from .llama_macro import RMSNorm, SwiGLUMLP, LlamaMIRASLayer, ModularLlama
from .moneta import MonetaBlock, moneta_factory, build_moneta_llama

__all__ = [
    "RMSNorm",
    "SwiGLUMLP",
    "LlamaMIRASLayer",
    "ModularLlama",
    "MonetaBlock",
    "moneta_factory",
    "build_moneta_llama",
]

