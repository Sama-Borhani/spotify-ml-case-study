import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def main(data_path: str, out_md: str) -> None:
    df = pd.read_csv(data_path)
    if "collaboration" not in df.columns:
        raise SystemExit("Missing 'collaboration'. Run data_prep first.")

    y = df["collaboration"].astype(int)
    num = df.select_dtypes(include="number").drop(columns=["duration_ms"], errors="ignore")

    scaler = StandardScaler()
    X = scaler.fit_transform(num)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    acc = accuracy_score(yte, (proba >= 0.5))

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "a", encoding="utf-8") as f:
        f.write(f"\n### Logistic Regression\nAUC: {auc:.3f} | Acc: {acc:.3f}\n")
    print(f" Logistic → AUC={auc:.3f} Acc={acc:.3f} (appended to {out_md})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Logistic Regression baseline for collaboration prediction.")
    p.add_argument("--data", required=True, help="Path to cleaned CSV")
    p.add_argument("--out", default="reports/results.md", help="Markdown file to append metrics")
    args = p.parse_args()
    main(args.data, args.out)
