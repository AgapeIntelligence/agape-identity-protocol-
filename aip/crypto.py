# aip/crypto.py
"""
Cryptographic utilities for Agape Identity Protocol (AIP) v1.0+

Derives deterministic, standards-compliant secp256k1 (Bitcoin-style) keypairs
directly from the RIS feature vector — turning the Resonant Individual State
into a usable wallet/private identity primitive.

Fully reproducible: same RIS → same keys forever.
"""

from __future__ import annotations

import hashlib
from typing import Tuple

import ecdsa
from ecdsa import SigningKey, SECP256k1
import base58
import numpy as np


def ris_to_private_key(
    feature_vector: np.ndarray,
    salt: bytes | None = None,
) -> bytes:
    """
    Deterministically derive a 32-byte secp256k1 private key from the RIS feature vector.

    Uses SHA3-512 → first 32 bytes (standard practice, same strength as Keccak-256 → 32).
    Optional salt for future domain separation if multiple keys are needed from one RIS.
    """
    data = feature_vector.tobytes()
    if salt:
        data = salt + data
    return hashlib.sha3_512(data).digest()[:32]


def private_key_to_wif(private_key_bytes: bytes, compressed: bool = True) -> str:
    """
    Convert raw 32-byte private key to Wallet Import Format (WIF).
    Default: compressed public key (Bitcoin mainnet prefix 0x80).
    """
    prefix = b"\x80"  # mainnet
    extended = prefix + private_key_bytes
    if compressed:
        extended += b"\x01"
    return base58.b58encode_check(extended).decode()


def private_key_to_public_hex(private_key_bytes: bytes, compressed: bool = True) -> str:
    """
    Derive compressed (33-byte) public key in hexadecimal.
    """
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.verifying_key

    x = vk.pubkey.point.x()
    y_parity = vk.pubkey.point.y() & 1
    prefix = b"\x02" if y_parity == 0 else b"\x03"
    return (prefix + x.to_bytes(32, "big")).hex()


def derive_crypto_from_ris(ris_state: dict) -> dict:
    """
    High-level helper used by examples and applications.
    """
    priv_bytes = ris_to_private_key(ris_state["feature_vector"])

    return {
        "private_key_bytes": priv_bytes,
        "private_key_wif": private_key_to_wif(priv_bytes),
        "public_key_hex": private_key_to_public_hex(priv_bytes),
        "address_legacy": base58.b58encode_check(
            b"\x00" + hashlib.new("ripemd160", hashlib.sha256(bytes.fromhex(
                private_key_to_public_hex(priv_bytes, compressed=True)
            )).digest()).digest()
        ).decode(),  # P2PKH – optional, shown for completeness
    }


# ——— Example ———
if __name__ == "__main__":
    from aip.core import SignalEmbedNet, NodeGraphNet, generate_RIS

    embed_net = SignalEmbedNet()
    graph_net = NodeGraphNet()
    ris = generate_RIS(graph_net, embed_net)

    keys = derive_crypto_from_ris(ris)

    print("\nPersonal Crypto Derived from RIS (v1.0 canonical)")
    print("=" * 60)
    print(f"Private Key (WIF)   : {keys['private_key_wif']}")
    print(f"Public Key (hex)    : {keys['public_key_hex']}")
    print(f"Legacy BTC Address  : {keys['address_legacy']}")
    print("\nThese keys are 100% deterministic — identical on every machine.")
