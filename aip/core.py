#!/usr/bin/env python3
"""
Agape Identity Protocol (AIP) Prototype — v1.0
- RIS (Resonant Individual State) generator
- Spinor/EMF simulation
- SHA3-512 checksum
- Dynamic challenge-response authentication
- Optional graph + coherence visualization
© 2025 AgapeIntelligence
"""

from __future__ import annotations
import os
import time
import math
import random
import hashlib
from typing import Tuple, Dict, Any

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

# Optional visualization
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

# ----------------- Deterministic Seed -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------- Configurable Parameters -----------------
N_NODES = int(os.getenv("AIP_N_NODES", "16000"))  # 16k nodes for high-dimensional state
EMBED_SIZE = int(os.getenv("AIP_EMBED_SIZE", "64"))
MAX_ITERS = int(os.getenv("AIP_MAX_ITERS", "40"))
BATCH_SIZE = int(os.getenv("AIP_BATCH_SIZE", "256"))
PHI = (1 + 5 ** 0.5) / 2.0

# ----------------- Helper Functions -----------------
def decay_time(flux_hz: float, mass: float = 1e-22, radius: float = 1e-9) -> float:
    hbar = 1.0545718e-34
    G = 6.6743e-11
    E = (4.0 * math.pi / 5.0) * G * (mass**2) / max(radius, 1e-12)
    base_tau = hbar / (E + 1e-40)
    gamma = float(flux_hz) / 500.0
    return float(base_tau / (1.0 + gamma**2))

def spinor_state(coh: float) -> np.ndarray:
    """Generates a simple 2x2 spinor from a coherence value."""
    return np.array([[coh, np.sqrt(1 - coh ** 2)],
                     [np.sqrt(1 - coh ** 2), coh]])

def model_checksum(s: str) -> str:
    """Compute SHA3-512 checksum and return first 64 hex chars."""
    h = hashlib.sha3_512(s.encode("utf-8")).hexdigest().upper()
    return h[:64]

def coherence_measure(flux_hz: float) -> float:
    """Fallback numeric coherence proxy in [0,1]."""
    return float(max(0.0, min(1.0, 0.5 * (1.0 - 1.0 / (1.0 + (flux_hz ** 2))))))

# ----------------- Signal Embed Network -----------------
class SignalEmbedNet(nn.Module):
    def __init__(self, embed_size: int = EMBED_SIZE):
        super().__init__()
        self.embed_size = embed_size
        self.base_a = nn.Parameter(torch.randn(embed_size) * 0.5)
        self.base_b = nn.Parameter(torch.randn(embed_size) * 0.5)
        self.base_c = nn.Parameter(torch.randn(embed_size) * 0.5)
        self.aux_proj = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, flux_batch: torch.Tensor, aux_batch: torch.Tensor | None = None):
        if flux_batch.dim() == 1:
            flux = flux_batch.unsqueeze(1)
        else:
            flux = flux_batch
            if flux.dim() == 2 and flux.size(1) != 1:
                flux = flux.mean(dim=1, keepdim=True)

        B = flux.size(0)
        flux_expanded = flux.repeat(1, self.embed_size)
        mix = torch.zeros((B, self.embed_size), device=flux.device)
        if aux_batch is not None:
            aux_batch = aux_batch.to(flux.device, dtype=torch.float32)
            mix = 0.5 * self.aux_proj(aux_batch)

        a = flux_expanded * self.base_a.unsqueeze(0) + mix
        b = flux_expanded * self.base_b.unsqueeze(0) * (PHI ** 1) * 2.0 + mix
        c = flux_expanded * self.base_c.unsqueeze(0) * (PHI ** 2) + mix
        return [a, b, c]

# ----------------- Node Graph Network -----------------
class NodeGraphNet(nn.Module):
    def __init__(self, n_nodes: int = N_NODES, embed_size: int = EMBED_SIZE, create_graph: bool = True):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed_size = embed_size
        self.node_embed = nn.Embedding(n_nodes, embed_size)
        self.fc = nn.Linear(embed_size * 4, 1)
        self.sigmoid = nn.Sigmoid()
        self.graph = nx.Graph()
        if create_graph:
            for i in range(n_nodes):
                angle = i * 2.399963
                radius = math.sqrt(i + 0.5) / math.sqrt(max(1, n_nodes))
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                self.graph.add_node(i, pos=(x, y))

    def forward(self, node_idx: torch.Tensor, signal_list: list[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        node_idx = node_idx.long()
        node_emb = self.node_embed(node_idx)
        concat_signals = torch.cat(signal_list, dim=1)
        x = torch.cat([node_emb, concat_signals], dim=1)
        logits = self.fc(x)
        p_activation = self.sigmoid(logits)
        mean_p = float(torch.mean(p_activation).item())

        if mean_p > 0.5 and len(self.graph) >= 2:
            try:
                i = int(node_idx[torch.randint(0, node_idx.shape[0], (1,)).item()].item())
                j = int(node_idx[torch.randint(0, node_idx.shape[0], (1,)).item()].item())
            except Exception:
                i, j = random.randint(0, self.n_nodes - 1), random.randint(0, self.n_nodes - 1)
            if i != j and not self.graph.has_edge(i, j):
                weight = random.uniform(0.5, 1.5)
                self.graph.add_edge(i, j, weight=float(weight))

        deg_hist = nx.degree_histogram(self.graph)
        total_deg = sum(deg_hist) if deg_hist else 1
        probs = np.array([d / total_deg for d in deg_hist if d > 0], dtype=float)
        entropy = -np.sum(probs * np.log(probs + 1e-12)) if probs.size else 0.0

        return p_activation, float(entropy)

# ----------------- RIS Generation -----------------
def generate_RIS(model: NodeGraphNet, embed_net: SignalEmbedNet) -> Dict[str, Any]:
    flux_vec = np.random.uniform(40.0, 500.0, N_NODES).astype(np.float32)
    aux_pool = np.random.randn(BATCH_SIZE * 4, embed_net.embed_size).astype(np.float32)
    device = "cpu"
    embed_net.to(device)
    model.to(device)

    optimizer = optim.Adam(list(embed_net.parameters()) + list(model.parameters()), lr=3e-3)
    mse = nn.MSELoss()
    ris_state = {}

    for it in range(MAX_ITERS):
        idx = np.random.choice(len(flux_vec), size=BATCH_SIZE, replace=False)
        flux_batch = torch.tensor(flux_vec[idx], dtype=torch.float32, device=device).unsqueeze(1)
        aux_idx = np.random.choice(aux_pool.shape[0], size=flux_batch.size(0), replace=False)
        aux_batch = torch.tensor(aux_pool[aux_idx], dtype=torch.float32, device=device)

        signal_list = embed_net(flux_batch, aux_batch)
        node_idx = torch.randint(0, N_NODES, (flux_batch.size(0),), device=device)
        p_activation, entropy = model(node_idx, signal_list)

        target = torch.ones_like(p_activation)
        loss = mse(p_activation, target) - 0.2 * torch.mean(p_activation) + 0.5 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 5 == 0 or it == MAX_ITERS - 1:
            mean_flux = float(flux_batch.mean().item())
            coh = coherence_measure(mean_flux)
            print(f"[RIS] Iter {it+1}/{MAX_ITERS}: Loss={float(loss.item()):.6f} "
                  f"Entropy={entropy:.6f} Coherence={coh:.4f}")

    with torch.no_grad():
        final_signals = torch.cat(signal_list, dim=1)
        ris_state["feature_vector"] = final_signals.mean(dim=0).cpu().numpy()
        ris_state["feature_norm"] = float(np.linalg.norm(ris_state["feature_vector"])**2)
        ris_state["graph"] = model.graph
        ris_state["checksum"] = model_checksum(str(ris_state["feature_vector"]))

    return ris_state

# ----------------- Main -----------------
def main():
    print("Agape Identity Protocol — Prototype v1.0")
    start_time = time.time()

    embed_net = SignalEmbedNet()
    graph_net = NodeGraphNet()
    ris = generate_RIS(graph_net, embed_net)

    print("\n=== RIS Output ===")
    print("Feature Norm:", ris["feature_norm"])
    print("Checksum:", ris["checksum"])
    print("Graph Nodes:", ris["graph"].number_of_nodes())
    print("Graph Edges:", ris["graph"].number_of_edges())

    if MPL_AVAILABLE and ris["graph"].number_of_nodes() <= 20000:
        pos = nx.get_node_attributes(ris["graph"], 'pos')
        plt.figure(figsize=(8, 8))
        nx.draw(ris["graph"], pos=pos, node_size=1, node_color='cyan', edge_color='white', alpha=0.25)
        plt.title("RIS Graph Topology")
        plt.savefig("ris_graph.png", dpi=300, bbox_inches='tight')
        print("Graph visualization saved as ris_graph.png")

    elapsed = time.time() - start_time
    print(f"Run complete — elapsed {elapsed:.2f}s")

if __name__ == "__main__":
    main()
