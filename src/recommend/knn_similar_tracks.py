import argparse
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def main(data_path: str, track_name: str, k: int, out_md: str) -> None:
    df = pd.read_csv(data_path)

    if "track_name" not in df.columns:
        raise SystemExit("Missing 'track_name' in data.")
    idx = df[df["track_name"].str.lower() == track_name.lower()].index
    if len(idx) == 0:
        raise SystemExit(f"Track '{track_name}' not found.")
    i = idx[0]

    feat = df.select_dtypes(include="number")
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(feat)
    _, indices = knn.kneighbors([feat.iloc[i].values])

    neighbors = df.iloc[indices[0][1:]]  # skip self
    artist_col = "artists" if "artists" in df.columns else None

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "a", encoding="utf-8") as f:
        title = df.loc[i, "track_name"]
        artist = df.loc[i, artist_col] if artist_col else "(unknown artist)"
        f.write(f"\n### KNN Similar Tracks for '{title}' by {artist}\n")
        for _, row in neighbors.iterrows():
            name = row["track_name"]
            arts = row[artist_col] if artist_col else "(unknown)"
            f.write(f"- {name} — {arts}\n")
    print(f" Wrote {k} similar tracks to {out_md}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Find k similar tracks using audio features.")
    p.add_argument("--data", required=True, help="Path to cleaned CSV")
    p.add_argument("--track", required=True, help="Track name to query")
    p.add_argument("--k", type=int, default=10, help="Number of neighbors")
    p.add_argument("--out", default="reports/results.md", help="Markdown file to append output")
    args = p.parse_args()
    main(args.data, args.track, args.k, args.out)
