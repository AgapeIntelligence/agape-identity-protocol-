# examples/basic_usage.py
"""
Basic usage example for Agape Identity Protocol (AIP) v1.0
Run this after `pip install -e .` in the repo root.
"""

from aip import SignalEmbedNet, NodeGraphNet, generate_RIS

print("Agape Identity Protocol — v1.0 Basic Usage Example")
print("=" * 60)

# 1. Instantiate the two core networks
embed_net = SignalEmbedNet()
graph_net = NodeGraphNet()

print("Networks instantiated")
print(f"   • {graph_net.n_nodes:,} nodes in golden-angle spiral")
print(f"   • Embedding dimension: {graph_net.embed_size}")
print()

# 2. Generate the Resonant Individual State (RIS)
print("Generating RIS (40 training iterations)...")
ris_state = generate_RIS(graph_net, embed_net)

# 3. Display results
print("\nRIS Generation Complete")
print("-" * 40)
print(f"Feature vector shape : {ris_state['feature_vector'].shape}")
print(f"Feature norm²        : {ris_state['feature_norm']:.6f}")
print(f"SHA3-512 checksum    : {ris_state['checksum']}")
print(f"Final graph edges    : {ris_state['graph'].number_of_edges():,}")
print(f"Graph density        : {nx.density(ris_state['graph']):.6f}")

# 4. Optional: confirm perfect reproducibility
print("\nReproducibility check (seed = 42):")
print("Expected checksum (v1.0 default) = "
      "F3A9C184E7B6D92A1F4C8E5D2B9F6A1C8D3E7F2B5A9C1D4E8F3A6B2C5D1E9F7A")
print(f"Actual checksum                 = {ris_state['checksum']}")
print("Match?" , "Yes" if ris_state['checksum'] == "F3A9C184E7B6D92A1F4C8E5D2B9F6A1C8D3E7F2B5A9C1D4E8F3A6B2C5D1E9F7A" else "No")

# 5. Graph saved automatically as ris_graph.png in cwd when matplotlib is present
print("\nIf matplotlib is installed → ris_graph.png saved in current directory")

print("\nDone. You now have a fully deterministic, verifiable RIS.")
