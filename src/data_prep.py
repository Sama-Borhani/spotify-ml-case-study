import argparse
import os
import pandas as pd

def main(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path)

    # Drop duplicate tracks if track_id exists
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"]).copy()

    # duration in minutes
    if "duration_ms" in df.columns:
        df["duration_mins"] = df["duration_ms"] / 60000.0

    # collaboration flag if multiple artists separated by ';'
    if "artists" in df.columns:
        df["collaboration"] = df["artists"].astype(str).str.contains(";", na=False)

    if "speechiness" in df.columns:
        df = df[df["speechiness"] < 0.66].copy()

    required = [c for c in ["artists", "album_name", "track_name"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned data → {out_path} (rows={len(df)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean + feature-engineer dataset (CSV → CSV).")
    p.add_argument("--in", dest="in_path", required=True, help="Path to raw CSV")
    p.add_argument("--out", dest="out_path", default="data/processed/clean.csv", help="Path to cleaned CSV")
    args = p.parse_args()
    main(args.in_path, args.out_path)
