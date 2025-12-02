import numpy as np
import pandas as pd
from html import unescape

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

INP = r"C:\Users\afaan\grants_clustered_chips_only.csv"
OUT = r"C:\Users\afaan\grants_clustered_chips_only_with_subclusters.csv"

df = pd.read_csv(INP)
df["title"] = df["title"].fillna("").astype(str).str.strip().map(unescape)

# Only cluster within the big cluster
big = df[df["cluster"] == 0].copy().reset_index(drop=True)
rest = df[df["cluster"] != 0].copy()

CUSTOM_STOP = {"ndash", "amp", "fy", "round", "program", "programs", "initiative", "initiatives", "office"}
stop = ENGLISH_STOP_WORDS.union(CUSTOM_STOP)

vec = TfidfVectorizer(stop_words=list(stop), ngram_range=(1, 2), min_df=2, max_df=0.85)
X = normalize(vec.fit_transform(big["title"]))

rng = np.random.default_rng(42)
n = X.shape[0]
sample_idx = rng.choice(n, size=min(300, n), replace=False)

ks = range(2, 7)
sil = {}
for k in ks:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=30, batch_size=128, max_iter=400)
    labels = km.fit_predict(X)
    sil[k] = float(silhouette_score(X[sample_idx], labels[sample_idx], metric="cosine"))

best_k = max(sil, key=sil.get)
print("Subcluster silhouette:", {k: round(v, 3) for k, v in sil.items()})
print("Chosen sub-k:", best_k)

km = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=50, batch_size=128, max_iter=600)
big["subcluster0"] = km.fit_predict(X)

# Add empty subcluster column for the rest
rest["subcluster0"] = -1

out = pd.concat([big, rest], ignore_index=True)
out.to_csv(OUT, index=False, encoding="utf-8")
print("Saved:", OUT)
print(out.groupby(["cluster", "subcluster0"]).size())
