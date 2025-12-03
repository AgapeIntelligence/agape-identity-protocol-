# aip/__init__.py
"""
Agape Identity Protocol (AIP) — public API
"""

from .core import (
    generate_RIS,
    SignalEmbedNet,
    NodeGraphNet,
    coherence_measure,
    model_checksum,
)

from .crypto import (
    ris_to_private_key,
    private_key_to_wif,
    private_key_to_public_hex,
    derive_crypto_from_ris,
)

__version__ = "1.0.0"

__all__ = [
    # Core RIS generation
    "generate_RIS",
    "SignalEmbedNet",
    "NodeGraphNet",
    "coherence_measure",
    "model_checksum",

    # Cryptographic derivation
    "ris_to_private_key",
    "private_key_to_wif",
    "private_key_to_public_hex",
    "derive_crypto_from_ris",
]

from .identity import derive_identity_from_ris
__all__ += ["derive_identity_from_ris"]
