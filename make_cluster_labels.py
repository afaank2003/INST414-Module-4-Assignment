import pandas as pd

INP = r"C:\Users\afaan\cluster_summary_chips_only.csv"
OUT = r"C:\Users\afaan\cluster_labels.csv"

summary = pd.read_csv(INP)

# Manually assign labels based on your printed summary
label_map = {
    0: "Industrial capacity / advanced manufacturing (broad bucket)",
    1: "Solid-state lighting manufacturing R&D (DOE manufacturing rounds)",
    2: "CHIPS Incentives Program (fabrication + materials/equipment facilities)",
    3: "Apprenticeship expansion (state workforce pipeline)",
    4: "Advanced manufacturing + power semiconductors (SiC packaging / general AM)",
    5: "Semiconductor / CHIPS technology protection (security, safeguarding, partner capabilities)",
}

summary["cluster_label"] = summary["cluster"].map(label_map).fillna("Unlabeled")

# Reorder columns for Medium readability
cols = ["cluster", "cluster_label", "size", "top_terms", "example_1", "example_2"]
summary = summary[cols].sort_values("cluster")

summary.to_csv(OUT, index=False, encoding="utf-8")
print("Saved:", OUT)
print(summary.to_string(index=False))
