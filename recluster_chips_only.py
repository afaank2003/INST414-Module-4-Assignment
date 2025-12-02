import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

INP = r"C:\Users\afaan\grants_clustered_filtered.csv"
OUT = r"C:\Users\afaan\grants_clustered_chips_only.csv"
SUMMARY_OUT = r"C:\Users\afaan\cluster_summary_chips_only.csv"

df = pd.read_csv(INP)
df["title"] = df["title"].fillna("").astype(str).str.strip()

# Drop obviously off-domain education-disability program clusters by keyword
bad = re.compile(r"(?:OSERS|OSEP|special education|rehabilitative|disabilities|Assistance Listing Number 84\.327)",
                 flags=re.IGNORECASE)
df = df[~df["title"].str.contains(bad, na=False, regex=True)].copy()

df = df.reset_index(drop=True)
print("Rows after dropping off-domain:", len(df))

# TF-IDF
vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)
X = normalize(vec.fit_transform(df["title"]))

# pick k again (small range)
rng = np.random.default_rng(42)
n = X.shape[0]
sample_size = min(400, n)
sample_idx = rng.choice(n, size=sample_size, replace=False)

ks = range(2, 8)
sil = {}
for k in ks:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=20, max_iter=300)
    labels = km.fit_predict(X)
    sil[k] = float(silhouette_score(X[sample_idx], labels[sample_idx], metric="cosine"))

best_k = max(sil, key=sil.get)
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
print("Chosen k:", best_k)

km = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=256, n_init=50, max_iter=500)
df["cluster"] = km.fit_predict(X)

df.to_csv(OUT, index=False, encoding="utf-8")
print("Saved:", OUT)
print("Cluster sizes:\n", df["cluster"].value_counts().sort_index())

# Build a new summary (top terms + 2 examples)
terms = np.array(vec.get_feature_names_out())

rows = []
for c in sorted(df["cluster"].unique()):
    idx = np.where(df["cluster"].values == c)[0]
    mean_vec = np.asarray(X[idx].mean(axis=0)).ravel()
    top = terms[np.argsort(-mean_vec)[:12]]
    ex = df[df["cluster"] == c].head(2)["title"].tolist()
    rows.append({
        "cluster": c,
        "size": int(len(idx)),
        "top_terms": ", ".join(top),
        "example_1": ex[0] if len(ex) > 0 else "",
        "example_2": ex[1] if len(ex) > 1 else "",
    })

summary = pd.DataFrame(rows).sort_values("cluster")
summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8")
print("Saved:", SUMMARY_OUT)
print(summary.to_string(index=False))
