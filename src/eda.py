import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    "danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_mins"
]

def main(data_path: str, out_dir: str = "reports/figures") -> None:
    df = pd.read_csv(data_path)
    os.makedirs(out_dir, exist_ok=True)

    for col in [c for c in FEATURES if c in df.columns]:
        plt.figure()
        df[col].dropna().hist(bins=40)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col); plt.ylabel("Count")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"hist_{col}.png")
        plt.savefig(out_path)
        plt.close()
    print(f"📊 Saved histograms → {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick EDA histograms.")
    p.add_argument("--data", required=True, help="Path to cleaned CSV")
    args = p.parse_args()
    main(args.data)
