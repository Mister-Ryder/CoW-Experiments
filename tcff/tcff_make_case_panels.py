"""Render preregistered failure-case panels from saved predictions and maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from topcow_dataset import TopCoWTestDataset


DATASETS = {
    "TopCoW_val": ("crown", "crown_processed"),
    "Lausanne": ("lausanne", "lausanne_processed"),
    "IXI_HH": ("ixi", "ixi_processed"),
}
CHARCOAL = "#25313C"


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def resize_to(array: np.ndarray, shape) -> np.ndarray:
    array = np.squeeze(array)
    if tuple(array.shape) == tuple(shape):
        return array
    factors = [target / source for target, source in zip(shape, array.shape)]
    return ndimage.zoom(array, factors, order=1)


def overlay_label(ax, image, label, title):
    low, high = np.quantile(image, [0.01, 0.995])
    ax.imshow(image, cmap="gray", vmin=low, vmax=high)
    masked = np.ma.masked_where(label <= 0, label)
    ax.imshow(masked, cmap="tab20", vmin=1, vmax=13, alpha=0.78,
              interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def choose_slice(gt, no_aux, residual, target):
    vessel = (gt > 0) | (no_aux > 0) | (residual > 0) | (target > 0)
    intervention_difference = (residual != no_aux) | (target != no_aux)
    prediction_error = (target != gt)
    score = (vessel.sum(axis=(1, 2)) +
             3 * intervention_difference.sum(axis=(1, 2)) +
             2 * prediction_error.sum(axis=(1, 2)))
    return int(np.argmax(score)), score


def main(args):
    project_root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    selection = pd.read_csv(
        results / "08_failure_cases" / "predeclared_case_selection.csv")
    selection = selection[selection["availability"].eq("selected")]
    output_dir = results / "08_failure_cases" / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for dataset_name, (dataset_tag, directory) in DATASETS.items():
        selected = selection[selection["dataset"].eq(dataset_name)]
        if selected.empty:
            continue
        dataset = TopCoWTestDataset(str(project_root / directory), split="test")
        samples = {sample["case_name"]: sample for sample in dataset.samples}
        run_dir = (results / "03_raw_runs" / "P0_G_SELECTED_CASE_MAPS" /
                   f"full_selected_maps_{dataset_tag}")
        metrics_path = run_dir / "case_metrics.csv"
        metrics = pd.read_csv(metrics_path).set_index("case_id")
        for case_id in sorted(selected["case_id"].dropna().astype(str).unique()):
            if case_id not in samples:
                index_rows.append({"dataset": dataset_name, "case_id": case_id,
                                   "status": "missing_dataset_case"})
                continue
            prediction_path = run_dir / "predictions" / f"{case_id}.npz"
            map_path = run_dir / "maps" / f"{case_id}.npz"
            if not prediction_path.exists() or not map_path.exists():
                index_rows.append({"dataset": dataset_name, "case_id": case_id,
                                   "status": "missing_prediction_or_map"})
                continue
            prediction = np.load(prediction_path)
            maps = np.load(map_path)
            image = np.load(samples[case_id]["image_path"]).astype(np.float32)
            gt = prediction["gt"]
            no_aux = prediction["no_aux"]
            residual = prediction["residual"]
            target = prediction["target"]
            slice_index, slice_scores = choose_slice(gt, no_aux, residual, target)
            t_map = resize_to(maps["shallow_T"], gt.shape)
            g_map = resize_to(maps["shallow_G"], gt.shape)
            u_value = float(np.asarray(maps["shallow_U"]).mean())
            utility = ((residual == gt).astype(np.int8) -
                       (no_aux == gt).astype(np.int8))
            reasons = selected[selected["case_id"].eq(case_id)].sort_values(
                ["selection_rule", "rank"])
            reason_text = "; ".join(reasons["selection_rule"].str.replace("_", " ").tolist())
            metric = metrics.loc[case_id]

            plt.rcParams.update({
                "font.family": "DejaVu Sans", "font.size": 8.5,
                "text.color": CHARCOAL, "axes.titlecolor": CHARCOAL,
                "figure.facecolor": "white", "axes.facecolor": "white",
            })
            fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.0))
            low, high = np.quantile(image[slice_index], [0.01, 0.995])
            axes[0, 0].imshow(image[slice_index], cmap="gray", vmin=low, vmax=high)
            axes[0, 0].set_title("MRA")
            axes[0, 0].axis("off")
            overlay_label(axes[0, 1], image[slice_index], gt[slice_index], "Ground truth")
            overlay_label(axes[0, 2], image[slice_index], no_aux[slice_index], "No auxiliary")
            overlay_label(axes[0, 3], image[slice_index], residual[slice_index],
                          "Uncalibrated residual")
            overlay_label(axes[1, 0], image[slice_index], target[slice_index], "TCFF")
            trust_image = axes[1, 1].imshow(t_map[slice_index], cmap="cividis", vmin=0, vmax=1)
            axes[1, 1].set_title(f"Executed trust T (U={u_value:.3f})")
            axes[1, 1].axis("off")
            fig.colorbar(trust_image, ax=axes[1, 1], fraction=0.046, pad=0.03)
            compatibility_image = axes[1, 2].imshow(
                g_map[slice_index], cmap="cividis", vmin=0, vmax=1)
            axes[1, 2].set_title("Spatial compatibility G")
            axes[1, 2].axis("off")
            fig.colorbar(compatibility_image, ax=axes[1, 2], fraction=0.046, pad=0.03)
            utility_image = axes[1, 3].imshow(
                utility[slice_index], cmap="PuOr", vmin=-1, vmax=1,
                interpolation="nearest")
            axes[1, 3].set_title("Voxel utility: residual − no auxiliary")
            axes[1, 3].axis("off")
            cbar = fig.colorbar(utility_image, ax=axes[1, 3], fraction=0.046, pad=0.03,
                                ticks=[-1, 0, 1])
            cbar.ax.set_yticklabels(["harm", "same", "benefit"])
            fig.suptitle(
                f"{dataset_name} · {case_id} · slice {slice_index} | "
                f"Dice={metric['dice']:.3f}, clDice={metric['cldice']:.3f}, "
                f"Det-F1={metric['det_f1']:.3f}", fontsize=11.5, y=0.985)
            fig.text(0.01, 0.02, f"Selection: {reason_text}", fontsize=7.5)
            fig.text(0.99, 0.02,
                     "Slice chosen by a fixed vessel + intervention-disagreement + TCFF-error score.",
                     ha="right", fontsize=7.5, color="#59636D")
            fig.tight_layout(rect=(0, 0.05, 1, 0.95))
            stem = f"{dataset_tag}__{case_id}"
            png_path = output_dir / f"{stem}.png"
            pdf_path = output_dir / f"{stem}.pdf"
            fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
            fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            index_rows.append({
                "dataset": dataset_name, "case_id": case_id,
                "status": "success", "slice_index": slice_index,
                "slice_selection_score": float(slice_scores[slice_index]),
                "selection_reasons": reason_text,
                "dice": float(metric["dice"]), "cldice": float(metric["cldice"]),
                "det_f1": float(metric["det_f1"]),
                "shallow_U": u_value,
                "png": png_path.relative_to(results).as_posix(),
                "pdf": pdf_path.relative_to(results).as_posix(),
                "prediction_source": prediction_path.relative_to(results).as_posix(),
                "map_source": map_path.relative_to(results).as_posix(),
            })
    index = pd.DataFrame(index_rows)
    atomic_csv(results / "08_failure_cases" / "case_panel_index.csv", index)
    print(index["status"].value_counts(dropna=False).to_json())
    return 0 if not index.empty and index["status"].eq("success").all() else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument("--results_root", default=str(here / "results_tcff_rebuttal"))
    raise SystemExit(main(parser.parse_args()))
