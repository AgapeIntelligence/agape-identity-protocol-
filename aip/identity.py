#!/usr/bin/env python3
"""
AIP Identity Layer — v1.0
Deterministic cryptographic identity system built on the Resonant Individual State (RIS).
Features:
• Hardened 64-byte seed from RIS feature vector
• secp256k1 (Bitcoin-compatible) keypair + WIF
• Ed25519 signing keypair (DID + challenge-response)
• DID method: did:aip:<base58-multihash>
• Challenge-response signing/verification
"""

from __future__ import annotations

import base58
import hashlib
import json
from typing import Dict, Any

import ecdsa
from ecdsa import SECP256k1, SigningKey
from nacl.signing import SigningKey as EdSigningKey
from nacl.signing import VerifyKey
import numpy as np


# ------------------------------------------------------------------
# 1. RIS → 64-byte cryptographically strong seed
# ------------------------------------------------------------------
def ris_to_seed(ris_state: Dict[str, Any], salt: bytes = b"AIP_v1") -> bytes:
    """
    SHA3-512 over feature_vector + optional salt.
    Output: 64 secure bytes suitable as BIP32/BIP39 root or direct key material.
    """
    data = ris_state["feature_vector"].tobytes()
    h = hashlib.sha3_512()
    h.update(data)
    h.update(salt)
    return h.digest()  # 64 bytes


# ------------------------------------------------------------------
# 2. secp256k1 keypair (Bitcoin-style)
# ------------------------------------------------------------------
def derive_secp256k1(seed: bytes) -> Dict[str, str]:
    priv_bytes = seed[:32]
    sk = SigningKey.from_string(priv_bytes, curve=SECP256k1)
    vk = sk.verifying_key

    # Compressed public key (33 bytes → hex)
    x = vk.pubkey.point.x().to_bytes(32, "big")
    prefix = b"\x02" if vk.pubkey.point.y() % 2 == 0 else b"\x03"
    pub_compressed = (prefix + x).hex()

    # WIF (compressed)
    wif = base58.b58encode_check(b"\x80" + priv_bytes + b"\x01").decode()

    return {
        "private_wif": wif,
        "public_hex": pub_compressed,
        "private_hex": priv_bytes.hex(),
    }


# ------------------------------------------------------------------
# 3. Ed25519 keypair (used for DID + signing)
# ------------------------------------------------------------------
def derive_ed25519(seed: bytes) -> Dict[str, str]:
    sk = EdSigningKey(seed[:32])
    vk = sk.verify_key
    return {
        "private_b64": sk.encode().hex(),           # hex for easier storage
        "public_b64": vk.encode().hex(),
        "private_seed": seed[:32].hex(),           # raw 32-byte seed
    }


# ------------------------------------------------------------------
# 4. DID document (method: did:aip:<base58>)
# ------------------------------------------------------------------
def build_did_document(ed25519_pub_hex: str) -> Dict[str, Any]:
    public_b64 = bytes.fromhex(ed25519_pub_hex).decode("ascii")  # 32 → b64
    did = f"did:aip:{base58.b58encode(bytes.fromhex(ed25519_pub_hex)).decode()[:32]}"

    return {
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": did,
        "verificationMethod": [{
            "id": f"{did}#key-1",
            "type": "Ed25519VerificationKey2020",
            "controller": did,
            "publicKeyBase64": public_b64,
        }],
        "authentication": [f"{did}#key-1"],
        "assertionMethod": [f"{did}#key-1"],
    }


# ------------------------------------------------------------------
# 5. Challenge–Response (Ed25519)
# ------------------------------------------------------------------
def sign_challenge(private_seed_hex: str, message: str) -> str:
    sk = EdSigningKey(bytes.fromhex(private_seed_hex))
    signed = sk.sign(message.encode("utf-8"))
    return signed.signature.hex()


def verify_challenge(public_hex: str, message: str, signature_hex: str) -> bool:
    vk = VerifyKey(bytes.fromhex(public_hex))
    try:
        vk.verify(message.encode("utf-8"), bytes.fromhex(signature_hex))
        return True
    except Exception:
        return False


# ------------------------------------------------------------------
# 6. Master wrapper — one call to rule them all
# ------------------------------------------------------------------
def derive_identity_from_ris(ris_state: Dict[str, Any]) -> Dict[str, Any]:
    seed = ris_to_seed(ris_state)

    secp = derive_secp256k1(seed)
    ed = derive_ed25519(seed)
    did_doc = build_did_document(ed["public_b64"])

    return {
        "seed_hex": seed.hex(),
        "secp256k1": secp,
        "ed25519": ed,
        "did_document": did_doc,
        "did": did_doc["id"],
    }


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    from aip.core import SignalEmbedNet, NodeGraphNet, generate_RIS

    embed_net = SignalEmbedNet()
    graph_net = NodeGraphNet()
    ris = generate_RIS(graph_net, embed_net)

    identity = derive_identity_from_ris(ris)

    print("\nAIP Identity Layer — Canonical v1.0 Output")
    print("=" * 60)
    print(f"DID            : {identity['did']}")
    print(f"secp256k1 WIF  : {identity['secp256k1']['private_wif']}")
    print(f"Ed25519 pub    : {identity['ed25519']['public_b64']}")
    print("\nChallenge → Response example:")
    challenge = "Hello, I am the same person as yesterday."
    sig = sign_challenge(identity['ed25519']['private_seed'], challenge)
    valid = verify_challenge(identity['ed25519']['public_b64'], challenge, sig)
    print(f"Signature valid? → {valid}")
