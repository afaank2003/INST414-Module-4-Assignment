import pandas as pd

NODES = r"C:\Users\afaan\chips_nodes.csv"
EDGES = r"C:\Users\afaan\chips_edges.csv"
CLUST = r"C:\Users\afaan\grants_clustered_chips_only.csv"

nodes = pd.read_csv(NODES)
edges = pd.read_csv(EDGES)
df = pd.read_csv(CLUST)

print("Validation checks")
print("-" * 60)

print("Nodes:", len(nodes))
print("Edges:", len(edges))

# Range checks
assert nodes["node_id"].min() == 0
assert nodes["node_id"].max() == len(nodes) - 1

# No self-loops
self_loops = (edges["source_id"] == edges["target_id"]).sum()
print("Self-loops:", int(self_loops))

# Similarity sanity
print("Cosine similarity range:", float(edges["cosine_similarity"].min()), "to", float(edges["cosine_similarity"].max()))

# Degree summary (treat as directed edges; fine for sanity checks)
out_deg = edges.groupby("source_id").size()
in_deg = edges.groupby("target_id").size()
deg = out_deg.add(in_deg, fill_value=0)

print("Degree stats (out+in):")
print(deg.describe())

# Cluster distribution
df["cluster"] = df["cluster"].astype(int)
print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

# Spot-check: show a few highest-degree nodes (often “generic” titles)
top = deg.sort_values(ascending=False).head(10).index
print("\nTop-degree nodes (often generic connectors):")
print(nodes.loc[nodes["node_id"].isin(top), ["node_id","title","cluster"]].sort_values("node_id").to_string(index=False))

print("\nOK: basic checks passed if no assertion error occurred.")
