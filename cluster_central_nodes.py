import pandas as pd

NODES = r"C:\Users\afaan\chips_nodes.csv"
EDGES = r"C:\Users\afaan\chips_edges.csv"
OUT = r"C:\Users\afaan\cluster_central_nodes.csv"

nodes = pd.read_csv(NODES)
edges = pd.read_csv(EDGES)

# Degree 
deg_out = edges.groupby("source_id").size()
deg_in = edges.groupby("target_id").size()
deg = deg_out.add(deg_in, fill_value=0).rename("degree")

nodes = nodes.merge(deg, left_on="node_id", right_index=True, how="left")
nodes["degree"] = nodes["degree"].fillna(0).astype(int)

# Top 3 most-connected nodes per cluster
top = (nodes.sort_values(["cluster", "degree"], ascending=[True, False])
            .groupby("cluster")
            .head(3)
            [["cluster", "node_id", "degree", "title"]])

top.to_csv(OUT, index=False, encoding="utf-8")

print("Saved:", OUT)
print(top.to_string(index=False))
