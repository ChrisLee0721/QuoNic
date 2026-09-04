"""Exp13 Analysis: Generate statistics and figures for paper.

Combines EC2 (99 circuits) and local (760 circuits) data.
Outputs:
  - exp13_stats.json: all computed metrics
  - fig_scaling_curves.png
  - fig_heatmap.png
  - fig_backend_frequency.png
  - exp13_outliers.csv: overhead > 5x circuits
  - exp13_full_table.csv: all 793 circuits
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────

def load_local() -> list[dict]:
    path = RESULTS_DIR / "exp13_local_checkpoint.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data.get("results", []):
        if "error" in r:
            continue
        rec = r.get("recommendation", {})
        fastest = r.get("fastest", {})
        rows.append({
            "circuit": r["circuit"],
            "n_qubits": r["n_qubits"],
            "gate_count": r.get("gate_count", 0),
            "depth": r.get("depth", 0),
            "decision_class": r.get("decision_class", "unknown"),
            "family": r.get("meta", {}).get("family", "unknown"),
            "depth_bucket": r.get("meta", {}).get("depth_bucket", ""),
            "rec_backend": rec.get("backend", ""),
            "rec_method": rec.get("method", ""),
            "rec_key": f"{rec.get('backend','')}/{rec.get('method','')}",
            "fastest_key": fastest.get("key", ""),
            "is_fastest": r.get("scheduler_is_fastest", False),
            "overhead": r.get("overhead_vs_fastest", 0),
            "rec_time": r.get("recommended_time", 0),
            "fastest_time": fastest.get("time", 0),
            "source": "local",
        })
    return rows


def load_ec2() -> list[dict]:
    path = RESULTS_DIR / "ec2_results.txt"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            m = re.search(
                r'Done:\s+(.+?):\s+(\S+)\s+vs\s+(\S+)\s+\|\s+(YES|NO)\s+\|\s+([\d.]+)x',
                line,
            )
            if not m:
                continue
            name, rec, fastest, match, speedup = m.groups()
            # Extract n_qubits from name
            nums = re.findall(r'-(\d+)', name)
            nq = int(nums[0]) if nums else 0
            # Guess family
            family = "unknown"
            for fam in ["GHZ", "QFT", "QPE", "QAOA", "VQE", "Grover", "LinCluster"]:
                if fam in name:
                    family = fam
                    break
            if "RandCliff" in name:
                family = "RandomClifford"
            elif "RandNonCliff" in name:
                family = "RandomNonClifford"
            # Guess depth_bucket
            db = ""
            for d in ["shallow", "medium", "deep"]:
                if d in name:
                    db = d
                    break
            # Guess decision_class
            dclass = "general"
            if family in ("GHZ", "Grover", "RandomClifford", "LinCluster"):
                dclass = "clifford"
            elif family in ("QPE", "QAOA", "VQE", "RandomNonClifford"):
                dclass = "low_tw"
            rows.append({
                "circuit": name,
                "n_qubits": nq,
                "gate_count": 0,
                "depth": 0,
                "decision_class": dclass,
                "family": family,
                "depth_bucket": db,
                "rec_backend": rec.split("/")[0] if "/" in rec else rec,
                "rec_method": rec.split("/")[1] if "/" in rec else "",
                "rec_key": rec,
                "fastest_key": fastest,
                "is_fastest": match == "YES",
                "overhead": float(speedup),
                "rec_time": 0,
                "fastest_time": 0,
                "source": "ec2",
            })
    return rows


def load_all() -> pd.DataFrame:
    local = load_local()
    ec2 = load_ec2()
    # Deduplicate: prefer local over ec2 (more detailed)
    seen = {r["circuit"] for r in local}
    combined = local + [r for r in ec2 if r["circuit"] not in seen]
    df = pd.DataFrame(combined)
    df = df[df["overhead"] > 0].copy()
    return df


# ── Statistics ─────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict:
    overheads = df["overhead"].values
    n = len(overheads)
    sorted_ov = np.sort(overheads)
    match_count = df["is_fastest"].sum()

    stats = {
        "n_circuits": int(n),
        "n_match": int(match_count),
        "match_rate_pct": round(100 * match_count / n, 1),
        "overhead_mean": round(float(np.mean(overheads)), 2),
        "overhead_median": round(float(np.median(overheads)), 2),
        "overhead_p95": round(float(np.percentile(overheads, 95)), 2),
        "overhead_worst": round(float(np.max(overheads)), 2),
        "overhead_geomean": round(float(math.exp(np.mean(np.log(overheads)))), 2),
    }

    # Per-class breakdown
    by_class = {}
    for cls in df["decision_class"].unique():
        sub = df[df["decision_class"] == cls]
        ov = sub["overhead"].values
        by_class[cls] = {
            "n_circuits": int(len(sub)),
            "match_rate_pct": round(100 * sub["is_fastest"].sum() / len(sub), 1),
            "overhead_mean": round(float(np.mean(ov)), 2),
            "overhead_median": round(float(np.median(ov)), 2),
            "overhead_p95": round(float(np.percentile(ov, 95)), 2),
            "overhead_worst": round(float(np.max(ov)), 2),
            "overhead_geomean": round(float(math.exp(np.mean(np.log(ov)))), 2),
        }
    stats["per_class"] = by_class

    # Per-size breakdown
    size_bins = [(4, 8, "small"), (10, 16, "medium"), (20, 24, "large")]
    by_size = {}
    for lo, hi, label in size_bins:
        sub = df[(df["n_qubits"] >= lo) & (df["n_qubits"] <= hi)]
        if len(sub) == 0:
            continue
        ov = sub["overhead"].values
        by_size[label] = {
            "n_circuits": int(len(sub)),
            "match_rate_pct": round(100 * sub["is_fastest"].sum() / len(sub), 1),
            "overhead_mean": round(float(np.mean(ov)), 2),
            "overhead_median": round(float(np.median(ov)), 2),
            "overhead_p95": round(float(np.percentile(ov, 95)), 2),
            "overhead_worst": round(float(np.max(ov)), 2),
        }
    stats["per_size"] = by_size

    return stats


# ── Figures ────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "clifford": "#2196F3",
    "general": "#4CAF50",
    "low_tw": "#FF9800",
}
CLASS_LABELS = {
    "clifford": "Clifford",
    "general": "General",
    "low_tw": "Low treewidth",
}


def fig_scaling_curves(df: pd.DataFrame):
    """Scaling curves: x=qubit count, y=overhead, lines by decision class."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in ["clifford", "general", "low_tw"]:
        sub = df[df["decision_class"] == cls].copy()
        if len(sub) == 0:
            continue
        # Group by n_qubits, compute median overhead
        grouped = sub.groupby("n_qubits")["overhead"].agg(["median", "mean", "max", "count"]).reset_index()
        ax.plot(grouped["n_qubits"], grouped["median"], "o-",
                color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], markersize=6, linewidth=2)
        # Shade between median and max
        ax.fill_between(grouped["n_qubits"], grouped["median"], grouped["max"],
                        alpha=0.15, color=CLASS_COLORS[cls])

    ax.set_xlabel("Qubit count", fontsize=12)
    ax.set_ylabel("Overhead vs fastest (×)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Scheduler Overhead by Circuit Class", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df["n_qubits"].unique()))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_scaling_curves.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig_scaling_curves.png")


def fig_heatmap(df: pd.DataFrame):
    """Heatmap: x=qubit count, y=circuit family, color=median overhead."""
    # Only use families with enough data
    family_counts = df["family"].value_counts()
    families = family_counts[family_counts >= 3].index.tolist()
    sub = df[df["family"].isin(families)]

    pivot = sub.pivot_table(values="overhead", index="family", columns="n_qubits", aggfunc="median")
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    # Sort families by median overhead
    pivot["_sort"] = pivot.median(axis=1)
    pivot = pivot.sort_values("_sort").drop(columns=["_sort"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    # Log scale colormap
    vmin = max(0.9, pivot.min().min())
    vmax = pivot.max().max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Qubit count", fontsize=12)
    ax.set_title("Median Overhead Heatmap (log scale)", fontsize=14)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 10 else "black"
                ax.text(j, i, f"{val:.1f}×", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Overhead (×)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_heatmap.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig_heatmap.png")


def fig_backend_frequency(df: pd.DataFrame):
    """Backend selection frequency: bar chart of scheduler recommendations."""
    # Count by rec_key
    counts = df["rec_key"].value_counts().head(12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    ax1.barh(range(len(counts)), counts.values, color=colors)
    ax1.set_yticks(range(len(counts)))
    ax1.set_yticklabels(counts.index, fontsize=9)
    ax1.set_xlabel("Number of circuits", fontsize=12)
    ax1.set_title("Scheduler Backend Selection", fontsize=14)
    ax1.invert_yaxis()

    # Pie chart for top 6
    top6 = counts.head(6)
    other = counts.iloc[6:].sum()
    if other > 0:
        top6 = pd.concat([top6, pd.Series({"other": other})])
    ax2.pie(top6.values, labels=top6.index, autopct="%1.1f%%", startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(top6))))
    ax2.set_title("Top Backend Share", fontsize=14)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_backend_frequency.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig_backend_frequency.png")


def fig_class_comparison(df: pd.DataFrame):
    """Box plot: overhead distribution by decision class."""
    fig, ax = plt.subplots(figsize=(7, 5))

    classes = ["clifford", "general", "low_tw"]
    data = [df[df["decision_class"] == c]["overhead"].values for c in classes]
    labels = [CLASS_LABELS[c] for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Overhead vs fastest (×)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Overhead Distribution by Circuit Class", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_class_comparison.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig_class_comparison.png")


# ── Outlier analysis ───────────────────────────────────────────────────────

def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outliers = df[df["overhead"] > 5].sort_values("overhead", ascending=False)
    return outliers


# ── Crossover analysis ─────────────────────────────────────────────────────

def analyze_crossover(df: pd.DataFrame) -> dict:
    """Check if scheduler correctly switches from SV to stabilizer at n≈20."""
    cliff = df[df["decision_class"] == "clifford"].copy()
    results = {}
    for n in sorted(cliff["n_qubits"].unique()):
        sub = cliff[cliff["n_qubits"] == n]
        recs = sub["rec_key"].value_counts().to_dict()
        results[int(n)] = {
            "n_circuits": len(sub),
            "recommendations": recs,
        }
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Exp13 Analysis")
    print("=" * 60)

    df = load_all()
    print(f"\nLoaded {len(df)} circuits")
    print(f"  Local: {(df['source']=='local').sum()}")
    print(f"  EC2:   {(df['source']=='ec2').sum()}")
    print(f"  Qubit range: {df['n_qubits'].min()}-{df['n_qubits'].max()}")

    # ── Statistics ──
    print("\n[Computing statistics...]")
    stats = compute_stats(df)
    print(f"\n  Overall:")
    print(f"    Circuits:      {stats['n_circuits']}")
    print(f"    Match rate:    {stats['match_rate_pct']}%")
    print(f"    Overhead:      mean={stats['overhead_mean']}x  median={stats['overhead_median']}x  "
          f"P95={stats['overhead_p95']}x  worst={stats['overhead_worst']}x")
    print(f"    Geometric mean: {stats['overhead_geomean']}x")

    print(f"\n  Per class:")
    for cls, info in stats["per_class"].items():
        print(f"    {cls:12s}: {info['n_circuits']:4d} circuits, "
              f"match={info['match_rate_pct']}%, "
              f"mean={info['overhead_mean']}x, P95={info['overhead_p95']}x, "
              f"worst={info['overhead_worst']}x")

    print(f"\n  Per size:")
    for size, info in stats["per_size"].items():
        print(f"    {size:8s}: {info['n_circuits']:4d} circuits, "
              f"match={info['match_rate_pct']}%, "
              f"mean={info['overhead_mean']}x, P95={info['overhead_p95']}x")

    # ── Figures ──
    print("\n[Generating figures...]")
    fig_scaling_curves(df)
    fig_heatmap(df)
    fig_backend_frequency(df)
    fig_class_comparison(df)

    # ── Outliers ──
    print("\n[Outlier analysis (overhead > 5x)...]")
    outliers = analyze_outliers(df)
    print(f"  Found {len(outliers)} outliers")
    if len(outliers) > 0:
        print(f"  Top 10:")
        for _, row in outliers.head(10).iterrows():
            print(f"    {row['circuit']:40s} {row['rec_key']:30s} vs {row['fastest_key']:30s} {row['overhead']:.1f}x")

    # ── Crossover ──
    print("\n[Crossover analysis (Clifford class)...]")
    crossover = analyze_crossover(df)
    for n, info in sorted(crossover.items()):
        recs_str = ", ".join(f"{k}:{v}" for k, v in sorted(info["recommendations"].items(), key=lambda x: -x[1]))
        print(f"  n={n:2d}: {info['n_circuits']:3d} circuits → {recs_str}")

    # ── Save outputs ──
    print("\n[Saving outputs...]")

    # Stats JSON
    stats_path = RESULTS_DIR / "exp13_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {stats_path.name}")

    # Full table CSV
    table_cols = ["circuit", "n_qubits", "family", "depth_bucket", "decision_class",
                  "rec_key", "fastest_key", "is_fastest", "overhead", "source"]
    df[table_cols].to_csv(RESULTS_DIR / "exp13_full_table.csv", index=False)
    print(f"  Saved exp13_full_table.csv")

    # Outliers CSV
    if len(outliers) > 0:
        outliers[table_cols].to_csv(RESULTS_DIR / "exp13_outliers.csv", index=False)
        print(f"  Saved exp13_outliers.csv ({len(outliers)} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()
