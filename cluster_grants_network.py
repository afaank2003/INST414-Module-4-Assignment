import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


PATH = r"C:\Users\afaan\grants_opps_clean.csv"

# --- 1) Load ---
df = pd.read_csv(PATH)

df["title"] = df["title"].fillna("").astype(str).str.strip()

# Parse open_date if present
if "open_date" in df.columns:
    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce")

# --- 2) Filter to CHIPS-relevant subset ---

# A) Time filter: broaden to include more opportunities
if "open_date" in df.columns:
    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce")
    df = df[df["open_date"] >= "2008-01-01"].copy()

# B) Keyword filter (non-capturing group avoids the warning)
chips_terms = re.compile(
    r"(?:"
    r"semiconductor|microelectronic|micro-electronic|chip\b|chips\b|"
    r"advanced packaging|packaging|wafer|fab\b|foundr|lithograph|"
    r"electronics manufacturing|manufacturing|industrial|"
    r"materials|metrology|nist\b|"
    r"supply chain|"
    r"workforce|apprenticeship|technician|"
    r"quantum|photonics"
    r")",
    flags=re.IGNORECASE
)

df = df[df["title"].str.contains(chips_terms, na=False, regex=True)].copy()

# Drop empty / short titles
df = df[df["title"].str.len() >= 15].copy()
df = df.drop_duplicates(subset=["opp_number"], keep="first") if "opp_number" in df.columns else df
df = df.reset_index(drop=True)

print("Rows after filters:", len(df))

# --- 3) TF-IDF ---
vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)
X = vec.fit_transform(df["title"])
X = normalize(X)
print("TF-IDF matrix:", X.shape)


# --- 4) Choose k ---
rng = np.random.default_rng(42)
n = X.shape[0]
sample_size = min(500, n)
sample_idx = rng.choice(n, size=sample_size, replace=False)

ks = range(2, 9)
sil = {}

for k in ks:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=10, max_iter=200)
    labels = km.fit_predict(X)
    sil[k] = float(silhouette_score(X[sample_idx], labels[sample_idx], metric="cosine"))

best_k = max(sil, key=sil.get)
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
print("Chosen k:", best_k)


# --- 5) Final clustering ---
km = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=256, n_init=20, max_iter=400)
df["cluster"] = km.fit_predict(X)

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())


# --- 6) Build kNN similarity network ---
K = 5
nn = NearestNeighbors(n_neighbors=min(K+1, n), metric="cosine")
nn.fit(X)
dist, idx = nn.kneighbors(X)

edges = []
for i in range(n):
    for r in range(1, min(K+1, n)):
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

# --- 7) Save outputs ---
df.to_csv(r"C:\Users\afaan\grants_clustered_filtered.csv", index=False, encoding="utf-8")
nodes_df.to_csv(r"C:\Users\afaan\grants_nodes_filtered.csv", index=False, encoding="utf-8")
edges_df.to_csv(r"C:\Users\afaan\grants_edges_filtered.csv", index=False, encoding="utf-8")

print("\nSaved:")
print(r"  C:\Users\afaan\grants_clustered_filtered.csv")
print(r"  C:\Users\afaan\grants_nodes_filtered.csv")
print(r"  C:\Users\afaan\grants_edges_filtered.csv")

# --- 8) Two examples per cluster ---
print("\nTwo examples per cluster:")
show_cols = ["cluster", "title"]
if "agency_name" in df.columns: show_cols.append("agency_name")
if "open_date" in df.columns: show_cols.append("open_date")
if "opp_number" in df.columns: show_cols.append("opp_number")
examples = df.sort_values(["cluster","open_date"], ascending=[True, False]).groupby("cluster").head(2)[show_cols]
print(examples.to_string(index=False))
