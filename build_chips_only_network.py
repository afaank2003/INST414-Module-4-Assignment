import numpy as np
import pandas as pd
from html import unescape

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

INP = r"C:\Users\afaan\grants_clustered_chips_only.csv"
NODES_OUT = r"C:\Users\afaan\chips_nodes.csv"
EDGES_OUT = r"C:\Users\afaan\chips_edges.csv"

K = 5

df = pd.read_csv(INP)
df["title"] = df["title"].fillna("").astype(str).str.strip().map(unescape)
df["title"] = (df["title"]
               .str.replace("&ndash;", " ", regex=False)
               .str.replace("&amp;", " ", regex=False))

CUSTOM_STOP = {"ndash", "amp", "fy", "round", "program", "programs", "initiative", "initiatives", "office", "announcement"}
stop = ENGLISH_STOP_WORDS.union(CUSTOM_STOP)

vec = TfidfVectorizer(stop_words=list(stop), ngram_range=(1, 2), min_df=2, max_df=0.85)
X = normalize(vec.fit_transform(df["title"]))

nn = NearestNeighbors(n_neighbors=min(K + 1, len(df)), metric="cosine")
nn.fit(X)
dist, idx = nn.kneighbors(X)

edges = []
n = len(df)
for i in range(n):
    for r in range(1, min(K + 1, n)):
        j = int(idx[i, r])
        sim = float(1.0 - dist[i, r])
        if i != j and sim > 0:
            edges.append((i, j, sim))

edges_df = pd.DataFrame(edges, columns=["source_id", "target_id", "cosine_similarity"]).drop_duplicates()

nodes_cols = ["title", "cluster"]
for c in ["opp_number", "agency_name", "agency_code", "open_date", "close_date", "opp_status", "aln_joined", "search_keyword"]:
    if c in df.columns:
        nodes_cols.append(c)

nodes_df = df[nodes_cols].copy()
nodes_df.insert(0, "node_id", np.arange(len(df)))

nodes_df.to_csv(NODES_OUT, index=False, encoding="utf-8")
edges_df.to_csv(EDGES_OUT, index=False, encoding="utf-8")

print("Saved:", NODES_OUT)
print("Saved:", EDGES_OUT)
print("Nodes:", len(nodes_df))
print("Edges:", len(edges_df))
