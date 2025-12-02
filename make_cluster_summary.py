import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

INP = r"C:\Users\afaan\grants_clustered_filtered.csv"
OUT = r"C:\Users\afaan\cluster_summary.csv"

df = pd.read_csv(INP)
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["cluster"] = df["cluster"].astype(int)

# Use the same basic text representation: titles -> TF-IDF -> cosine-friendly normalization
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
X = normalize(vec.fit_transform(df["title"]))
terms = np.array(vec.get_feature_names_out())

def top_terms(cluster_id: int, n_terms: int = 12):
    idx = np.where(df["cluster"].values == cluster_id)[0]
    mean_vec = np.asarray(X[idx].mean(axis=0)).ravel()
    top_idx = np.argsort(-mean_vec)[:n_terms]
    return [terms[i] for i in top_idx if mean_vec[i] > 0]

rows = []
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c].copy()
    if "open_date" in sub.columns:
        sub["open_date"] = pd.to_datetime(sub["open_date"], errors="coerce")
        sub = sub.sort_values("open_date", ascending=False)

    ex = sub.head(2)["title"].tolist()
    rows.append({
        "cluster": c,
        "size": int(len(sub)),
        "top_terms": ", ".join(top_terms(c)),
        "example_1": ex[0] if len(ex) > 0 else "",
        "example_2": ex[1] if len(ex) > 1 else "",
    })

summary = pd.DataFrame(rows).sort_values("cluster")
summary.to_csv(OUT, index=False, encoding="utf-8")

print("Saved:", OUT)
print(summary.to_string(index=False))
