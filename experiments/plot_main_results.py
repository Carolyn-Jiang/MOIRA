"""Generate the main paper figure families from explicitly selected, saved seeds.

No inference or data downloads. Batch ranking is handled by summarize_results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.main_artifacts import run_directory, write_json

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ["LONG_SHORT", "SHORT_LONG", "FLAT"]
STATES = ["HEALTHY", "WARNING", "BREAK"]


def load_run(path, modes):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"Incomplete/failed run: {path}")
    if not set(modes) <= set(manifest["sizing"]):
        raise ValueError(f"Requested sizing missing in {path}; available: {manifest['sizing']}")
    metrics = json.loads((path / "metrics.json").read_text())
    if len(metrics) != len(manifest["sizing"]) or {m["sizing"] for m in metrics} != set(manifest["sizing"]):
        raise ValueError(f"Missing or duplicate sizing metrics: {path}")
    if any(m.get(k) != manifest[k] for m in metrics for k in ("provider", "model", "seed")):
        raise ValueError(f"Metric identity disagrees with manifest: {path}")
    run = {"path": path, "manifest": manifest, "label": f"{manifest['model']} / seed {manifest['seed']}",
           "training": json.loads((path / "logs/training/direction_training_cycles.json").read_text()),
           "metrics": {m["sizing"]: m for m in metrics}}
    for mode in modes:
        logs = path / "logs/testing"
        equity = pd.read_csv(path / f"equity_{mode}.csv")
        comparison = json.loads((path / f"comparison_{mode}.json").read_text())
        main = next(r["strategy"] for r in comparison if r["group"] == "main")
        if list(equity["date"]) != manifest.get("valuation_dates", manifest["test_dates"]):
            raise ValueError(f"Equity dates disagree with manifest: {path} / {mode}")
        if not np.isclose(equity[main].iloc[-1], run["metrics"][mode]["final_value"]):
            raise ValueError(f"Equity endpoint disagrees with metrics: {path} / {mode}")
        if not np.isclose(equity[main].iloc[-1] - 1, run["metrics"][mode]["cumulative_return"]):
            raise ValueError(f"Cumulative return disagrees with equity: {path} / {mode}")
        run[mode] = {"equity": equity, "main": main,
                     "cf": pd.read_csv(logs / f"counterfactual_pnl_test_{mode}.csv"),
                     "decisions": pd.read_csv(logs / f"decision_log_{mode}.csv"),
                     "lifecycle": pd.read_csv(logs / f"lifecycle_log_{mode}.csv")}
    return run


def counterfactual_stats(frame):
    values = frame[["cf_long_short", "cf_short_long", "cf_flat"]].to_numpy(dtype=float)
    if not len(values):
        return None, np.array([])
    best = np.asarray(ACTIONS)[values.argmax(axis=1)]
    accuracy = float((frame["actual_action"].to_numpy() == best).mean())
    regret = values.max(axis=1) - frame["actual_pnl"].to_numpy(dtype=float)
    return accuracy, regret


def generate_figures(runs, modes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from experiments.figure_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans", "font.size": 10})

    if not runs:
        raise ValueError("Select at least one run")
    # Comparisons must share the same data/protocol; never silently align unrelated windows.
    for key in ("test_dates", "valuation_dates", "settlement_date", "train_dates", "dataset_hash", "input_hash", "pool_hash", "config"):
        if any(r["manifest"].get(key) != runs[0]["manifest"].get(key) for r in runs[1:]):
            raise ValueError(f"Selected runs disagree on {key}; plot them separately")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "qa").mkdir()
    labels = [r["label"] for r in runs]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(runs))))
    paths = []

    def save(fig, name):
        fig.tight_layout()
        # Colorbar axes are decorations rather than comparable data panels.
        colorbars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
        require_matplotlib_panel_alignment(
            fig, json_out=str(output / "qa" / f"{name}.alignment.json"),
            exclude_axes=colorbars, strict=True)
        for extension in ("png", "pdf", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    # Figures 2, 3, 5 and 10a: one row per explicitly chosen model/seed.
    for metric, name, title, scale in [
        ("cumulative_return", "final_return_comparison", "Final return (%)", 100),
        ("sharpe_ratio", "sharpe_comparison", "Annualized Sharpe", 1),
        ("max_drawdown", "max_drawdown_comparison", "Maximum drawdown (%)", 100),
        ("counterfactual_accuracy", "counterfactual_accuracy", "Strict CF accuracy (%)", 100),
    ]:
        fig, ax = plt.subplots(figsize=(12, max(3.5, len(runs)*.85+1.5)))
        y = np.arange(len(runs))
        width = 0.8 / len(modes)
        for i, mode in enumerate(modes):
            values = [r["metrics"][mode].get(metric) for r in runs]
            values = [v * scale if v is not None else np.nan for v in values]
            bars = ax.barh(y - .4 + width/2 + i*width, values, width, label=mode)
            ax.bar_label(bars, labels=[f"{v:.2f}" if np.isfinite(v) else "n/a" for v in values],
                         padding=4, fontsize=8)
        ax.margins(x=.3)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, name)

    # Figure 9: the shared No-Beta training occurs once per seed.
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, max(3, 2.8*len(runs))), squeeze=False)
    for ax, run in zip(axes.flat, runs):
        cycles = run["training"]
        x = [c["cycle"] for c in cycles]
        ax.plot(x, [c["reward_before"] for c in cycles], "o-", label="Before")
        ax.plot(x, [c["reward_after"] for c in cycles], "s--", label="Candidate")
        for c in cycles:
            ax.scatter(c["cycle"], c["reward_after"], marker="^" if c["prompt_accepted"] else "x",
                       c="green" if c["prompt_accepted"] else "red", s=70)
            if c.get("early_stop"):
                ax.axvline(c["cycle"], color="purple", ls=":")
        ax.set_title(run["label"])
        ax.set_xlabel("Training cycle")
        ax.set_ylabel("Reward")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "training_dynamics")

    for mode in modes:
        # Figures 1, 4, 10b. Equity is saved at every test session (includes day zero).
        for kind, title in [("cumulative_return", "Cumulative return (%)"),
                            ("drawdown_curves", "Drawdown (%)"),
                            ("counterfactual_regret", "Cumulative strict CF regret")]:
            fig, ax = plt.subplots(figsize=(12, 5))
            for color, run in zip(colors, runs):
                data = run[mode]
                if kind == "counterfactual_regret":
                    _, regret = counterfactual_stats(data["cf"])
                    y, dates = regret.cumsum(), data["cf"]["date"]
                else:
                    eq = data["equity"][data["main"]]
                    y = (eq - 1)*100 if kind == "cumulative_return" else (eq/eq.cummax()-1)*100
                    dates = data["equity"]["date"]
                ax.plot(pd.to_datetime(dates), y, label=run["label"], color=color)
            ax.set_title(f"{title} — {mode}")
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
            ax.grid(alpha=.2)
            fig.autofmt_xdate()
            save(fig, f"{kind}_{mode}")

        # Figure 8: use finalized decisions, matching the historical CF-based figures.
        fig, ax = plt.subplots(figsize=(12, max(3.5, len(runs)*.7+1.5)))
        left = np.zeros(len(runs))
        action_counts = []
        for action, color in zip(ACTIONS, ["#77A898", "#C28B83", "#C7CCD1"]):
            counts = np.array([(r[mode]["cf"]["actual_action"] == action).sum() for r in runs])
            ax.barh(range(len(runs)), counts, left=left, label=action, color=color)
            action_counts.append(counts)
            left += counts
        for i in range(len(runs)):
            counts_label = " / ".join(str(int(counts[i])) for counts in action_counts)
            ax.text(left[i] + max(1., left.max()*.02), i, counts_label, va="center", fontsize=9)
        ax.set_xlim(0, max(1., left.max())*1.4 + 2)
        ax.set_yticks(range(len(runs)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Finalized actions (right labels: Long–Short / Short–Long / Flat)")
        ax.set_title(f"Finalized action counts — {mode}")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, f"action_distribution_{mode}")

        # Figures 6, 7, 11: daily held pairs from this sizing's decision log.
        dates = runs[0]["manifest"]["test_dates"]
        sequences = [r[mode]["decisions"].set_index("date")["active_pair"].reindex(dates).fillna("NO_DATA")
                     for r in runs]
        pairs = sorted({p for seq in sequences for p in seq})
        pair_colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(pairs))))
        grid = np.array([[int((seq == p).sum()) for p in pairs] for seq in sequences])
        fig, ax = plt.subplots(figsize=(max(8, len(pairs)*1.2), max(3, len(runs)*.6+2)))
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pairs)), pairs, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(runs)), labels)
        for i, j in np.ndindex(grid.shape):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="Held sessions")
        ax.set_title(f"Pair selection heatmap — {mode}")
        save(fig, f"pair_selection_heatmap_{mode}")

        pair_index = {p: i for i, p in enumerate(pairs)}
        switches = []
        timeline = []
        for seq in sequences:
            timeline.append([pair_index.get(p, np.nan) for p in seq])
            switches.append(np.flatnonzero(seq.to_numpy()[1:] != seq.to_numpy()[:-1]) + 1)
        fig, ax = plt.subplots(figsize=(13, max(3, .65*len(runs)+2)))
        ax.imshow(timeline, cmap=ListedColormap(pair_colors[:len(pairs)]), aspect="auto",
                  vmin=-.5, vmax=len(pairs)-.5, interpolation="nearest")
        ax.set_yticks(range(len(runs)), labels)
        ax.set_xlabel("Test session index")
        ax.set_title(f"Pair-switch comparison — {mode}")
        ax.legend([Patch(color=pair_colors[i]) for i in range(len(pairs))], pairs,
                  loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        save(fig, f"pair_switch_comparison_{mode}")

        for i, run in enumerate(runs):
            data = run[mode]
            key = f"run_{i+1}_seed_{run['manifest']['seed']}_{mode}"
            fig, ax = plt.subplots(figsize=(13, max(3, .55*len(pairs)+2)))
            seq = timeline[i]
            start = 0
            for end in range(1, len(seq)+1):
                if end == len(seq) or seq[end] != seq[start]:
                    p = int(seq[start])
                    ax.barh(p, end-start, left=start, color=pair_colors[p])
                    ax.text(start+(end-start)/2, p, str(end-start), ha="center", va="center", fontsize=7)
                    start = end
            for switch in switches[i]:
                ax.axvline(switch, color="grey", ls="--", lw=.7)
            ax.set_yticks(range(len(pairs)), pairs)
            ax.set_xlabel("Test session index")
            ax.set_title(f"Pair timeline — {run['label']} — {mode} (labels: held sessions)")
            save(fig, f"pair_timeline_{key}")

            fig, ax = plt.subplots(figsize=(12, 5))
            eq = data["equity"]
            for name in eq.columns.drop("date"):
                ax.plot(pd.to_datetime(eq["date"]), (eq[name]-1)*100, label=name,
                        lw=2.5 if name == data["main"] else 1)
            ax.set_title(f"Main strategy and matched baselines — {run['label']} — {mode}")
            ax.set_ylabel("Return (%)")
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
            fig.autofmt_xdate()
            save(fig, f"baseline_comparison_{key}")

        # Figures 12–13: each pass has its own stored lifecycle, no reconstruction.
        life_grid = []
        fig, axes = plt.subplots(len(runs), 1, figsize=(12, max(3, 2.6*len(runs))), squeeze=False)
        for ax, run in zip(axes.flat, runs):
            life = run[mode]["lifecycle"].set_index("date").reindex(dates)
            life_grid.append(life["lifecycle_state"].map(dict(zip(STATES, range(3)))).to_numpy())
            score = pd.to_numeric(life["break_score"], errors="coerce")
            ax.plot(range(len(dates)), score, lw=1)
            validation = life["is_validation_day"].astype(str).str.lower().isin(["true", "1"])
            x = np.flatnonzero(validation)
            ax.scatter(x, score.iloc[x], c="purple", s=20, label="Validation day")
            ax.axhline(3, color="orange", ls="--")
            ax.axhline(5, color="red", ls="--")
            ax.set_title(run["label"])
            ax.set_ylabel("Break score")
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
        axes[-1, 0].set_xlabel("Test session index")
        save(fig, f"break_score_evolution_{mode}")
        fig, ax = plt.subplots(figsize=(13, max(3, .65*len(runs)+2)))
        cmap = ListedColormap(["#62b187", "#f0c35a", "#d65b5b"]).with_extremes(bad="#cccccc")
        ax.imshow(np.asarray(life_grid, dtype=float), aspect="auto", cmap=cmap, vmin=-.5, vmax=2.5,
                  interpolation="nearest")
        ax.set_yticks(range(len(runs)), labels)
        ax.set_xlabel("Test session index")
        ax.set_title(f"Lifecycle state — {mode}")
        ax.legend([Patch(color=c) for c in ["#62b187", "#f0c35a", "#d65b5b", "#cccccc"]],
                  [*STATES, "No data"], loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, f"lifecycle_state_timeline_{mode}")

    write_json(output / "selection.json", {"runs": [os.path.relpath(r["path"], output) for r in runs],
               "selected": [{k: r["manifest"][k] for k in ("provider", "model", "seed")} for r in runs],
               "sizing": modes, "figures": [p.name for p in paths]})
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--provider", default="open_source")
    parser.add_argument("--models", nargs="+", help="Exact served model IDs")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--select", nargs=3, action="append", metavar=("PROVIDER", "MODEL", "SEED"),
                        help="Repeat to compare chosen seeds across families, including closed models")
    parser.add_argument("--sizing", choices=["nobeta", "beta", "both"], default="both")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.select:
        if args.models or args.seeds:
            parser.error("Use either --select or --models with --seeds")
        try:
            selected = [(p, m, int(s)) for p, m, s in args.select]
        except ValueError:
            parser.error("Each --select SEED must be an integer")
    else:
        if not args.models or not args.seeds:
            parser.error("Explicit selection required: --models with --seeds, or --select PROVIDER MODEL SEED")
        selected = [(args.provider, m, s) for m in args.models for s in args.seeds]
    if len(set(selected)) != len(selected) or any(not 0 <= s < 2**32 for _, _, s in selected):
        parser.error("Selections must be distinct; seeds must be in [0, 2**32)")
    modes = ["nobeta", "beta"] if args.sizing == "both" else [args.sizing]
    paths = [run_directory(args.runs_dir, provider, model, seed) for provider, model, seed in selected]
    # Load every selected run before creating output, so missing seeds fail clearly.
    runs = [load_run(path, modes) for path in paths]
    for run, (provider, model, seed) in zip(runs, selected):
        if any(run["manifest"][k] != v for k, v in zip(("provider", "model", "seed"), (provider, model, seed))):
            raise ValueError(f"Selection disagrees with manifest: {run['path']}")
    digest = hashlib.sha256(json.dumps([str(p.resolve()) for p in paths] + modes).encode()).hexdigest()[:10]
    output = args.output_dir or args.runs_dir / "visualizations" / f"selection_{digest}"
    figures = generate_figures(runs, modes, output)
    print(f"Generated {len(figures)} figures from {selected}: {output}")
    return 0



def generate_sizing_comparisons(by_mode, output):
    """Compare separately selected Beta/No-Beta runs, displaying both seed IDs.

    These are descriptive differences between selected outcomes, not paired-seed
    causal estimates. The complete per-seed metrics remain in all_seeds.csv.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.figure_alignment import require_matplotlib_panel_alignment
    output = Path(output)
    (output / "qa").mkdir(exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10})
    modes = ["nobeta", "beta"]
    labels = [r["manifest"]["model"] for r in by_mode["nobeta"]]
    source = []
    for i, model in enumerate(labels):
        source.append({"model": model, **{mode: {"seed": by_mode[mode][i]["manifest"]["seed"],
            "metrics": by_mode[mode][i]["metrics"][mode]} for mode in modes}})
    write_json(output / "sizing_comparison.json", {"selection": "independent per sizing",
               "interpretation": "Differences between selected runs; not paired-seed effects", "models": source})

    def save(fig, name):
        fig.tight_layout()
        require_matplotlib_panel_alignment(fig, json_out=str(output / "qa" / f"{name}.alignment.json"), strict=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    for metric, name, xlabel, scale in [
            ("cumulative_return", "final_return_comparison", "Return (%)", 100),
            ("sharpe_ratio", "sharpe_comparison", "Annualized Sharpe", 1),
            ("max_drawdown", "max_drawdown_comparison", "Maximum drawdown (%)", 100),
            ("counterfactual_accuracy", "counterfactual_accuracy", "Strict CF accuracy (%)", 100)]:
        fig, ax = plt.subplots(figsize=(12, max(3.5, len(labels)*.85+1.5)))
        for j, mode in enumerate(modes):
            values = [r[mode]["metrics"].get(metric) for r in source]
            values = [v*scale if v is not None else np.nan for v in values]
            bars = ax.barh(np.arange(len(labels))+(j-.5)*.38, values, height=.36,
                           label="No-Beta" if mode == "nobeta" else "Beta", color=["#ABC9DC", "#3976A7"][j])
            ax.bar_label(bars, labels=[f"{v:.2f} (s{r[mode]['seed']})" if np.isfinite(v) else f"n/a (s{r[mode]['seed']})"
                         for v, r in zip(values, source)], padding=4, fontsize=8)
        ax.margins(x=.45)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title("Selected runs; seeds chosen separately for each sizing")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, name)
    for metric, name, xlabel, scale in [
            ("cumulative_return", "beta_effect_return", "Selected Beta − selected No-Beta return (pp)", 100),
            ("sharpe_ratio", "beta_effect_sharpe", "Selected Beta − selected No-Beta Sharpe", 1)]:
        values = [(r["beta"]["metrics"].get(metric), r["nobeta"]["metrics"].get(metric)) for r in source]
        values = [(b-n)*scale if b is not None and n is not None else np.nan for b, n in values]
        fig, ax = plt.subplots(figsize=(12, max(3.5, len(labels)*.6+1.5)))
        bars = ax.barh(range(len(labels)), values, color="#6E96A6")
        ax.bar_label(bars, labels=[f"{v:+.2f}" if np.isfinite(v) else "n/a" for v in values], padding=4, fontsize=9)
        ax.set_yticks(range(len(labels)), [f"{r['model']} (Beta s{r['beta']['seed']}; No-Beta s{r['nobeta']['seed']})" for r in source])
        ax.invert_yaxis()
        ax.margins(x=.3)
        ax.axvline(0, color="grey", lw=.6)
        ax.set_xlabel(xlabel)
        ax.set_title("Descriptive selected-run difference; not a paired-seed effect")
        save(fig, name)


if __name__ == "__main__":
    raise SystemExit(main())
