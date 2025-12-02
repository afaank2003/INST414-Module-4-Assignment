# CHIPS-Adjacent Grant Opportunities: Network Clustering (Grants.gov)

This project collects CHIPS-adjacent funding opportunities from Grants.gov, builds a similarity network using TF-IDF + cosine similarity, and clusters the opportunities to identify themes (workforce, CHIPS incentives, technology protection, etc.).

## Data Source
- Grants.gov search results collected via script (see `src/fetch_and_clean_grants.py`).
- The repo does not include raw full downloads by default (see `.gitignore`).

## Method
- Text representation: TF-IDF over opportunity titles (optionally enriched with metadata).
- Similarity: cosine similarity.
- Network: k-nearest-neighbor edges per opportunity.
- Clustering: k-means over TF-IDF vectors (k chosen by silhouette score).

## How to Run (Pipeline)
1. Fetch + clean:
   - `python src/fetch_and_clean_grants.py`
2. Filter/recluster to CHIPS-only:
   - `python src/recluster_chips_only.py`
3. Build network files (nodes/edges):
   - `python src/build_chips_only_network.py`
4. Summaries + labels:
   - `python src/make_cluster_summary.py`
   - `python src/make_cluster_labels.py`
5. Validate + representatives:
   - `python src/validate_network_and_clusters.py`
   - `python src/cluster_central_nodes.py`

## Outputs
Key deliverables saved to `outputs/`:
- `cluster_labels.csv`
- `cluster_summary_chips_only.csv`
- `cluster_central_nodes.csv`
- `chips_nodes.csv`, `chips_edges.csv`
