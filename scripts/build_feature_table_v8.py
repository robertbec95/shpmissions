"""Builds v8: v7's 41 columns, but with the 9 raw `banner_*` columns
replaced by 2 PCA-summarized components (`banner_PC1`/`banner_PC2`) -
exact same methodology as the original `PC1`/`PC2` POI summarization
(StandardScaler -> PCA(n_components=2)), applied this time to competitor
counts by CHAIN instead of by category.

This directly tests the hypothesis from v7's writeup: that the `banner`
group's failure wasn't about the underlying signal being useless, but
about feeding 9 raw sparse columns into a group whose total weight-pull
scales with column count. If PCA-summarizing fixes that, the group should
stop actively hurting once it's down to 2 columns like every other
PCA-summarized group in this pipeline.

Reproduce with: `PYTHONPATH=src python3 scripts/build_feature_table_v8.py`
"""

import json

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

V7_FILE = "data/location_features/store_feature_table_v7.json"
OUT_FILE = "data/location_features/store_feature_table_v8.json"

BANNER_COLS = [
    "banner_independent_other", "banner_mega_image", "banner_unnamed", "banner_carrefour", "banner_froo",
    "banner_la_doi_pasi", "banner_penny", "banner_auchan", "banner_annabella",
]


def main():
    v7 = json.load(open(V7_FILE))
    df = pd.DataFrame(v7).T
    banner_df = df[BANNER_COLS].astype(float)

    scaler = StandardScaler().fit(banner_df)
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(banner_df))
    pcs = pca.transform(scaler.transform(banner_df))
    print(f"Banner PCA explained variance ratio: {pca.explained_variance_ratio_}")

    v8 = {}
    for i, loc_id in enumerate(df.index):
        row = {k: v for k, v in v7[loc_id].items() if k not in BANNER_COLS}
        row["banner_PC1"] = float(pcs[i, 0])
        row["banner_PC2"] = float(pcs[i, 1])
        v8[loc_id] = row

    with open(OUT_FILE, "w") as f:
        json.dump(v8, f, indent=2)
    print(f"Wrote {OUT_FILE} ({len(next(iter(v8.values())))} columns)")
    for loc_id, row in v8.items():
        print(f"  {loc_id}: banner_PC1={row['banner_PC1']:.3f}, banner_PC2={row['banner_PC2']:.3f}")


if __name__ == "__main__":
    main()
