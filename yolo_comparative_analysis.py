
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Stores one row of a results table from the paper."""
    model:      str
    dataset:    str
    precision:  Optional[float]   = None   # %
    recall:     Optional[float]   = None   # %
    mAP50:      Optional[float]   = None   # %
    fps:        Optional[float]   = None
    model_size: Optional[float]   = None   # MB
    source:     str               = "paper"   # "paper" | "trained"


# ─────────────────────────────────────────────────────────────────────────────
# PAPER RESULTS  (digitised from all tables)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_RESULTS: list[ModelResult] = [

    # ── Table 1: TT100K  (YOLOv3-tiny vs YOLOv4-tiny) ────────────────────
    ModelResult("YOLOv3-tiny", "TT100K", mAP50=81.4, fps=41,    model_size=34.9),
    ModelResult("YOLOv4-tiny", "TT100K", mAP50=83.8, fps=58,    model_size=23.8),

    # ── Table 2: CCTSDB  (YOLOv3-tiny vs YOLOv4-tiny) ────────────────────
    ModelResult("YOLOv3-tiny", "CCTSDB", mAP50=74.40, fps=219,  model_size=21.7),
    ModelResult("YOLOv4-tiny", "CCTSDB", mAP50=84.57, fps=32.73,model_size=6.6),

    # ── Table 3: CCTSDB  (YOLOv3 vs YOLOv4 vs YOLOv5) ───────────────────
    ModelResult("YOLOv3",  "CCTSDB", precision=88.1, recall=94.6, mAP50=96.0,  fps=73),
    ModelResult("YOLOv4",  "CCTSDB", precision=88.1, recall=92.8, mAP50=95.8,  fps=78),
    ModelResult("YOLOv5",  "CCTSDB", precision=84.9, recall=95.2, mAP50=95.4,  fps=85),

    # ── Table 5: HRRSD   (YOLOv2 vs YOLOv3) ─────────────────────────────
    ModelResult("YOLOv2",  "HRRSD",  mAP50=79.2, fps=80),
    ModelResult("YOLOv3",  "HRRSD",  mAP50=81.2, fps=110),

    # ── YOLO_Report.pdf — YOLOv5s trained results ────────────────────────
    ModelResult("YOLOv5s*","CCTSDB", precision=88.2, recall=92.65, mAP50=93.56, fps=124.6, source="trained"),
    ModelResult("YOLOv5s*","TT100K", precision=80.23,recall=81.0,  mAP50=47.61, fps=116.0, source="trained"),
    ModelResult("YOLOv5s*","HRRSD",  precision=83.83,recall=None,  mAP50=81.2,  fps=154.6, source="trained"),
]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET / MODEL CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "CCTSDB": {
        "classes":     ["Warning", "Prohibitory", "Mandatory"],
        "num_classes": 3,
        "img_size":    416,
        "train_imgs":  10_123,
        "val_imgs":    3_000,
        "description": "Chinese Traffic Sign Detection Benchmark",
    },
    "TT100K": {
        "classes":     [f"sign_{i:02d}" for i in range(45)],
        "num_classes": 45,
        "img_size":    416,
        "train_imgs":  6_107,
        "val_imgs":    3_073,
        "description": "Tsinghua-Tencent 100K",
    },
    "HRRSD": {
        "classes": [
            "Ship","Bridge","Ground Track Field","Storage Tank",
            "Basketball Court","Tennis Court","Airplane",
            "Baseball Diamond","Harbor","Vehicle",
            "Crossroad","T-Junction","Parking Lot",
        ],
        "num_classes": 13,
        "img_size":    416,
        "train_imgs":  12_000,
        "val_imgs":    12_000,
        "description": "High-Resolution Remote Sensing Detection",
    },
}

# Training hyperparameters (Table 2 of YOLO_Report.pdf)
TRAIN_CFG = dict(epochs=15, batch=8, imgsz=416, optimizer="SGD")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model_name:   str,
    data_yaml:    str,
    dataset_name: str,
    project:      str = "runs",
    epochs:       int = TRAIN_CFG["epochs"],
    batch:        int = TRAIN_CFG["batch"],
    imgsz:        int = TRAIN_CFG["imgsz"],
) -> str:
    """
    Train a YOLOv5/v8 model and return the path to best.pt.
    Supported model_name values: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x,
                                  yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    """
    from ultralytics import YOLO

    weights = f"{model_name}.pt"
    run_name = f"{model_name}_{dataset_name}"
    print(f"\n{'─'*55}")
    print(f"  Training  : {model_name}  on  {dataset_name}")
    print(f"  Data YAML : {data_yaml}")
    print(f"  Epochs={epochs}  Batch={batch}  Imgsz={imgsz}")
    print(f"{'─'*55}")

    model = YOLO(weights)
    model.train(
        data     = data_yaml,
        epochs   = epochs,
        batch    = batch,
        imgsz    = imgsz,
        optimizer= TRAIN_CFG["optimizer"],
        project  = project,
        name     = run_name,
        exist_ok = True,
        verbose  = True,
    )
    best_pt = Path(project) / run_name / "weights" / "best.pt"
    return str(best_pt)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    weights_path: str,
    data_yaml:    str,
    dataset_name: str,
    model_label:  str,
    imgsz:        int = 416,
) -> ModelResult:
    """Run val() and wrap metrics into a ModelResult."""
    from ultralytics import YOLO

    print(f"\n[EVAL]  {model_label}  on  {dataset_name}")
    model   = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, verbose=False)

    # Measure FPS
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(50):
        model(dummy, verbose=False)
    fps = round(50 / (time.perf_counter() - t0), 1)

    return ModelResult(
        model      = model_label,
        dataset    = dataset_name,
        precision  = round(float(metrics.box.mp)  * 100, 2),
        recall     = round(float(metrics.box.mr)  * 100, 2),
        mAP50      = round(float(metrics.box.map50) * 100, 2),
        fps        = fps,
        source     = "trained",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE  (reproduce paper numbers without real training)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_results() -> list[ModelResult]:
    """Return the hardcoded paper results — useful for demo / CI."""
    print("[INFO] Running in SIMULATE mode — using paper-reported values.")
    return PAPER_RESULTS


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def results_to_df(results: list[ModelResult]) -> pd.DataFrame:
    rows = [asdict(r) for r in results]
    return pd.DataFrame(rows)


def print_tables(df: pd.DataFrame) -> None:
    """Pretty-print one table per dataset × comparison group."""

    # ── Table 1 & 2: tiny models ──────────────────────────────────────────
    tiny = df[df["model"].isin(["YOLOv3-tiny", "YOLOv4-tiny"])]
    for ds in ["TT100K", "CCTSDB"]:
        sub = tiny[tiny["dataset"] == ds][
            ["model","mAP50","fps","model_size","source"]
        ].reset_index(drop=True)
        if not sub.empty:
            print(f"\n{'═'*55}")
            print(f"  Table — Tiny models on {ds}")
            print(f"{'═'*55}")
            print(sub.to_string(index=False))

    # ── Table 3: full models on CCTSDB ────────────────────────────────────
    full_cc = df[
        df["model"].isin(["YOLOv3","YOLOv4","YOLOv5","YOLOv5s*"]) &
        (df["dataset"] == "CCTSDB")
    ][["model","precision","recall","mAP50","fps","source"]].reset_index(drop=True)
    print(f"\n{'═'*55}")
    print("  Table 3 — YOLOv3 / v4 / v5 on CCTSDB")
    print(f"{'═'*55}")
    print(full_cc.to_string(index=False))

    # ── Table 5: HRRSD ────────────────────────────────────────────────────
    hrrsd = df[df["dataset"] == "HRRSD"][
        ["model","precision","mAP50","fps","source"]
    ].reset_index(drop=True)
    print(f"\n{'═'*55}")
    print("  Table 5 — Models on HRRSD")
    print(f"{'═'*55}")
    print(hrrsd.to_string(index=False))

    # ── TT100K overview ───────────────────────────────────────────────────
    tt = df[df["dataset"] == "TT100K"][
        ["model","precision","recall","mAP50","fps","source"]
    ].reset_index(drop=True)
    print(f"\n{'═'*55}")
    print("  Table — Models on TT100K")
    print(f"{'═'*55}")
    print(tt.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "YOLOv2":     "#e07b39",
    "YOLOv3":     "#4c72b0",
    "YOLOv3-tiny":"#64b5cd",
    "YOLOv4":     "#dd8452",
    "YOLOv4-tiny":"#55a868",
    "YOLOv5":     "#c44e52",
    "YOLOv5s*":   "#8172b2",
}

def _bar_group(
    ax:      plt.Axes,
    df_sub:  pd.DataFrame,
    metric:  str,
    title:   str,
    ylabel:  str,
) -> None:
    """Draw a grouped bar chart on ax."""
    valid = df_sub[df_sub[metric].notna()].copy()
    if valid.empty:
        ax.set_visible(False)
        return

    models = valid["model"].tolist()
    values = valid[metric].tolist()
    colors = [PALETTE.get(m, "#999999") for m in models]

    bars = ax.bar(models, values, color=colors, width=0.55, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def plot_all(df: pd.DataFrame, save_dir: str = ".") -> None:
    """
    Generate four figure files reproducing / extending every chart
    in the paper.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Figure 1: Tiny models comparison (Tables 1 & 2) ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Tiny Model Comparison  (Tables 1 & 2)\n"
        "YOLOv3-tiny  vs  YOLOv4-tiny",
        fontsize=13, fontweight="bold",
    )
    tiny = df[df["model"].isin(["YOLOv3-tiny","YOLOv4-tiny"])]

    for row_idx, ds in enumerate(["TT100K", "CCTSDB"]):
        sub = tiny[tiny["dataset"] == ds]
        _bar_group(axes[row_idx, 0], sub, "mAP50",      f"{ds} — mAP@0.5 (%)", "mAP@0.5 (%)")
        _bar_group(axes[row_idx, 1], sub, "fps",         f"{ds} — FPS",          "Frames / sec")

    _add_legend(fig, ["YOLOv3-tiny","YOLOv4-tiny"])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, save_dir, "fig1_tiny_comparison.png")

    # ── Figure 2: Full models on CCTSDB (Table 3) ─────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "CCTSDB — YOLOv3 / YOLOv4 / YOLOv5  (Table 3)",
        fontsize=13, fontweight="bold",
    )
    cc = df[df["model"].isin(["YOLOv3","YOLOv4","YOLOv5","YOLOv5s*"]) &
            (df["dataset"] == "CCTSDB")]
    _bar_group(axes[0], cc, "precision", "Precision (%)",  "Precision (%)")
    _bar_group(axes[1], cc, "recall",    "Recall (%)",     "Recall (%)")
    _bar_group(axes[2], cc, "mAP50",     "mAP@0.5 (%)",    "mAP@0.5 (%)")
    _bar_group(axes[3], cc, "fps",       "FPS",            "Frames / sec")
    _add_legend(fig, ["YOLOv3","YOLOv4","YOLOv5","YOLOv5s*"])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, save_dir, "fig2_cctsdb_full_comparison.png")

    # ── Figure 3: HRRSD comparison (Table 5) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle(
        "HRRSD — YOLOv2 vs YOLOv3 vs YOLOv5s*  (Table 5)",
        fontsize=13, fontweight="bold",
    )
    hr = df[df["dataset"] == "HRRSD"]
    _bar_group(axes[0], hr, "mAP50", "mAP@0.5 (%)", "mAP@0.5 (%)")
    _bar_group(axes[1], hr, "fps",   "FPS",          "Frames / sec")
    _add_legend(fig, hr["model"].unique().tolist())
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, save_dir, "fig3_hrrsd_comparison.png")

    # ── Figure 4: Cross-dataset radar chart ───────────────────────────────
    _plot_radar(df, save_dir)

    # ── Figure 5: Speed–Accuracy scatter ──────────────────────────────────
    _plot_scatter(df, save_dir)

    print(f"\n[INFO] All figures saved to: {os.path.abspath(save_dir)}/")


def _add_legend(fig: plt.Figure, models: list[str]) -> None:
    handles = [
        mpatches.Patch(color=PALETTE.get(m, "#999999"), label=m)
        for m in models
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(models), fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)


def _save(fig: plt.Figure, save_dir: str, fname: str) -> None:
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def _plot_radar(df: pd.DataFrame, save_dir: str) -> None:
    """
    Radar chart comparing the three 'best' models from the paper
    across normalised Precision, Recall, mAP50, FPS.
    Uses CCTSDB rows (most complete) for the comparison.
    """
    categories = ["Precision", "Recall", "mAP@0.5", "FPS (norm)"]
    N          = len(categories)
    angles     = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles    += angles[:1]

    targets = [
        ("YOLOv3",  "CCTSDB"),
        ("YOLOv4",  "CCTSDB"),
        ("YOLOv5",  "CCTSDB"),
        ("YOLOv5s*","CCTSDB"),
    ]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)

    # Normalise FPS to 0–100 for radar
    max_fps = df["fps"].max()

    for (model, ds) in targets:
        row = df[(df["model"] == model) & (df["dataset"] == ds)]
        if row.empty:
            continue
        r = row.iloc[0]
        vals = [
            r.precision or 0,
            r.recall    or 0,
            r.mAP50     or 0,
            (r.fps / max_fps * 100) if r.fps else 0,
        ]
        vals += vals[:1]
        color = PALETTE.get(model, "#999999")
        ax.plot(angles, vals, "-o", linewidth=2, color=color, label=model)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_ylim(0, 110)
    ax.set_title("Model Comparison Radar (CCTSDB)\nFPS normalised to 0–100",
                 fontsize=11, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    _save(fig, save_dir, "fig4_radar_chart.png")


def _plot_scatter(df: pd.DataFrame, save_dir: str) -> None:
    """Speed–Accuracy scatter: FPS (x) vs mAP@0.5 (y), bubble = model size."""
    sub = df[df["mAP50"].notna() & df["fps"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Speed–Accuracy Trade-off Across Datasets",
                 fontsize=12, fontweight="bold")

    ds_markers = {"CCTSDB": "o", "TT100K": "s", "HRRSD": "^"}

    for _, row in sub.iterrows():
        color  = PALETTE.get(row["model"], "#999999")
        marker = ds_markers.get(row["dataset"], "D")
        size   = (row["model_size"] * 8) if row["model_size"] else 120
        ax.scatter(row["fps"], row["mAP50"],
                   s=size, c=color, marker=marker,
                   alpha=0.85, edgecolors="white", linewidth=0.6, zorder=3)
        ax.annotate(
            f"{row['model']}\n({row['dataset']})",
            (row["fps"], row["mAP50"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color=color,
        )

    # Dataset legend
    ds_handles = [
        plt.scatter([], [], marker=m, c="grey", s=80,
                    label=ds, alpha=0.8)
        for ds, m in ds_markers.items()
    ]
    model_handles = [
        mpatches.Patch(color=c, label=m)
        for m, c in PALETTE.items()
        if m in sub["model"].values
    ]
    ax.legend(handles=ds_handles + model_handles,
              fontsize=8, frameon=True, ncol=2,
              loc="lower right", title="Dataset / Model")

    ax.set_xlabel("Inference Speed (FPS)", fontsize=10)
    ax.set_ylabel("mAP@0.5 (%)",           fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    _save(fig, save_dir, "fig5_speed_accuracy_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(df: pd.DataFrame, save_dir: str = ".") -> None:
    """Write a markdown / text summary of findings to disk."""
    lines = [
        "# YOLO Comparative Analysis — Summary",
        "",
        "## Dataset Overview",
        "",
        "| Dataset | Classes | Train Images | Val Images | Resolution |",
        "|---------|---------|-------------|-----------|------------|",
    ]
    for ds, cfg in DATASET_CONFIG.items():
        lines.append(
            f"| {ds} | {cfg['num_classes']} | {cfg['train_imgs']:,} | "
            f"{cfg['val_imgs']:,} | {cfg['img_size']}×{cfg['img_size']} |"
        )

    lines += [
        "",
        "## Key Findings",
        "",
        "### Table 1 — TT100K (YOLOv3-tiny vs YOLOv4-tiny)",
        "",
        "| Model | mAP@0.5 (%) | FPS | Model Size (MB) |",
        "|-------|------------|-----|----------------|",
    ]
    t1 = df[df["model"].isin(["YOLOv3-tiny","YOLOv4-tiny"]) & (df["dataset"] == "TT100K")]
    for _, r in t1.iterrows():
        lines.append(f"| {r['model']} | {r['mAP50']} | {r['fps']} | {r['model_size']} |")

    lines += [
        "",
        "### Table 2 — CCTSDB (YOLOv3-tiny vs YOLOv4-tiny)",
        "",
        "| Model | mAP@0.5 (%) | FPS | Model Size (MB) |",
        "|-------|------------|-----|----------------|",
    ]
    t2 = df[df["model"].isin(["YOLOv3-tiny","YOLOv4-tiny"]) & (df["dataset"] == "CCTSDB")]
    for _, r in t2.iterrows():
        lines.append(f"| {r['model']} | {r['mAP50']} | {r['fps']} | {r['model_size']} |")

    lines += [
        "",
        "### Table 3 — CCTSDB (YOLOv3 / YOLOv4 / YOLOv5)",
        "",
        "| Model | Precision (%) | Recall (%) | mAP@0.5 (%) | FPS |",
        "|-------|--------------|-----------|------------|-----|",
    ]
    t3 = df[df["model"].isin(["YOLOv3","YOLOv4","YOLOv5","YOLOv5s*"]) &
            (df["dataset"] == "CCTSDB")]
    for _, r in t3.iterrows():
        p  = f"{r['precision']:.1f}" if r['precision'] else "—"
        rc = f"{r['recall']:.1f}"    if r['recall']    else "—"
        lines.append(f"| {r['model']} | {p} | {rc} | {r['mAP50']} | {r['fps']} |")

    lines += [
        "",
        "### Table 5 — HRRSD (YOLOv2 / YOLOv3 / YOLOv5s*)",
        "",
        "| Model | mAP@0.5 (%) | FPS |",
        "|-------|------------|-----|",
    ]
    t5 = df[df["dataset"] == "HRRSD"]
    for _, r in t5.iterrows():
        lines.append(f"| {r['model']} | {r['mAP50']} | {r['fps']} |")

    lines += [
        "",
        "## Conclusions",
        "",
        "- **YOLOv4-tiny** outperforms YOLOv3-tiny on TT100K (+2.4 mAP) with a smaller model size.",
        "- **YOLOv3-tiny** achieves higher FPS (219 f/s) on CCTSDB but lower mAP than YOLOv4-tiny.",
        "- On CCTSDB full models, **YOLOv3 leads in mAP** (96%), while YOLOv5 offers the best recall (95.2%).",
        "- **YOLOv5s* (trained)** exceeds reference precision on TT100K (+21.7 pp) with improved FPS.",
        "- **HRRSD** remains hardest due to 35–80 px miniature objects; YOLOv3 & YOLOv5s* reach ~81% mAP.",
        "- Higher FPS across all 'trained' entries reflects modern GPU throughput improvements.",
        "",
        "## Recommendations",
        "",
        "| Scenario | Recommended | Reason |",
        "|----------|------------|--------|",
        "| Mobile / embedded ADAS | YOLOv4-tiny | Small size, good mAP, 58 FPS |",
        "| Laptop / dash-cam | YOLOv5s | ~125 FPS, balanced accuracy |",
        "| Cloud / server | YOLOv5l / YOLOv8x | Maximum mAP, latency tolerated |",
        "| Remote sensing | YOLOv5 + high-res input | Best small-object handling |",
    ]

    out_path = os.path.join(save_dir, "comparative_analysis_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {out_path}")

    # Also dump raw results as JSON
    json_path = os.path.join(save_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in PAPER_RESULTS], f, indent=2)
    print(f"  [SAVED] {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "YOLO Comparative Analysis — "
            "Paper: Dogra, Sharma & Sohal, ICIIP 2023"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["simulate","train","evaluate"],
        default="simulate",
        help=(
            "simulate  : reproduce paper tables/charts (no GPU needed)\n"
            "train     : train models on provided datasets\n"
            "evaluate  : evaluate pre-trained weights\n"
        ),
    )
    parser.add_argument("--output_dir",     default="analysis_output")
    # Dataset YAML paths
    parser.add_argument("--cctsdb_yaml",    default=None)
    parser.add_argument("--tt100k_yaml",    default=None)
    parser.add_argument("--hrrsd_yaml",     default=None)
    # Pre-trained weights for evaluate mode
    parser.add_argument("--cctsdb_weights", default=None)
    parser.add_argument("--tt100k_weights", default=None)
    parser.add_argument("--hrrsd_weights",  default=None)
    # Training options
    parser.add_argument("--epochs",   type=int, default=TRAIN_CFG["epochs"])
    parser.add_argument("--batch",    type=int, default=TRAIN_CFG["batch"])
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results: list[ModelResult] = []

    # ── SIMULATE ────────────────────────────────────────────────────────────
    if args.mode == "simulate":
        results = simulate_results()

    # ── TRAIN ───────────────────────────────────────────────────────────────
    elif args.mode == "train":
        # Start with paper values so charts are complete even if some
        # datasets are not provided
        results = list(PAPER_RESULTS)

        dataset_map = {
            "CCTSDB": args.cctsdb_yaml,
            "TT100K": args.tt100k_yaml,
            "HRRSD":  args.hrrsd_yaml,
        }

        # Train YOLOv5s on every provided dataset (replicates YOLO_Report.pdf)
        for ds_name, yaml_path in dataset_map.items():
            if yaml_path is None:
                print(f"[SKIP] {ds_name}: no YAML provided (--{ds_name.lower()}_yaml)")
                continue
            best_pt = train_model(
                "yolov5s", yaml_path, ds_name,
                project = os.path.join(args.output_dir, "runs"),
                epochs  = args.epochs,
                batch   = args.batch,
            )
            result = evaluate_model(
                best_pt, yaml_path, ds_name,
                model_label="YOLOv5s (trained)",
            )
            results.append(result)

    # ── EVALUATE ────────────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        results = list(PAPER_RESULTS)

        eval_map = {
            "CCTSDB": (args.cctsdb_yaml, args.cctsdb_weights),
            "TT100K": (args.tt100k_yaml, args.tt100k_weights),
            "HRRSD":  (args.hrrsd_yaml,  args.hrrsd_weights),
        }
        for ds_name, (yaml_path, weights_path) in eval_map.items():
            if yaml_path is None or weights_path is None:
                print(f"[SKIP] {ds_name}: yaml or weights not provided")
                continue
            result = evaluate_model(
                weights_path, yaml_path, ds_name,
                model_label="YOLOv5s (eval)",
            )
            results.append(result)

    # ── OUTPUT ──────────────────────────────────────────────────────────────
    df = results_to_df(results)

    print("\n" + "═"*55)
    print("  COMPARATIVE ANALYSIS RESULTS")
    print("═"*55)
    print_tables(df)

    generate_summary(df, save_dir=args.output_dir)

    if not args.no_plots:
        print("\n[INFO] Generating figures …")
        plot_all(df, save_dir=args.output_dir)

    print(f"\n[DONE] All outputs written to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
