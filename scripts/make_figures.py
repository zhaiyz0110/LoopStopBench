"""Generate the paper figures and graphical abstract from frozen results.

The script reads CSV artifacts without running models or replaying trajectories.
``result/AUTHORITATIVE_RESULTS.md`` records the inputs and statistical decisions.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = WORKSPACE / "result"
DEFAULT_P0_RESULTS = WORKSPACE / "result"
DEFAULT_FIGURES = WORKSPACE / "reproduced" / "figures"
RESULTS = DEFAULT_RESULTS
P0_RESULTS = DEFAULT_P0_RESULTS

COLOR = {"code": "#2a78d6", "writing": "#eb6834", "QA": "#1baf7a"}
INK = "#151515"
MUTED = "#747474"
GRID = "#dedede"
LIGHT = "#d7d7d7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.facecolor": "white",
    }
)


CELL_SPECS = [
    ("code_8b", "m2_code_8b", "code 8B", "code"),
    ("code_32b", "m2_code_32b", "code 32B", "code"),
    ("writing_small", "m2_writing_small_judged", "writing small", "writing"),
    ("writing_mid", "m2_writing_mid_judged", "writing mid", "writing"),
    ("QA", "qa_m2", "QA", "QA"),
]
CELLS = [spec[0] for spec in CELL_SPECS]
FAMILY = {cell: family for cell, _, _, family in CELL_SPECS}
CELL_LABEL = {cell: label for cell, _, label, _ in CELL_SPECS}


def _read(name: str) -> list[dict[str, str]]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"required result is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required result is empty: {path}")
    return rows


def _read_p0(name: str) -> list[dict[str, str]]:
    path = P0_RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"required corrected P0 result is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"corrected P0 result is empty: {path}")
    return rows


def _row(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in match.items()):
            return row
    raise KeyError(f"no row matching {match}; available loops: {[r.get('loop') for r in rows]}")


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float("nan") if value in ("", None) else float(value)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.{{pdf,png}}")


def _asymmetric_error(point: float, lower: float, upper: float) -> np.ndarray:
    return np.array([[max(0.0, point - lower)], [max(0.0, upper - point)]])


def _forest_row(
    ax: plt.Axes,
    y: float,
    point: float,
    lower: float | None,
    upper: float | None,
    color: str,
) -> None:
    ax.scatter([point], [y], color=color, s=30, zorder=3)
    if lower is not None and upper is not None:
        ax.plot([lower, upper], [y, y], color=color, lw=1.6, zorder=2)
        for endpoint in (lower, upper):
            ax.plot([endpoint, endpoint], [y - 0.14, y + 0.14], color=color, lw=1.4)
    elif upper is not None:
        ax.annotate(
            "",
            xy=(upper, y),
            xytext=(point, y),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5},
            zorder=2,
        )


# Figure 1: primary regret decomposition.
def fig1_decomposition(out_dir: Path) -> None:
    primary = _read("oof_b1000_l005.csv")
    corrected_code = _read_p0("p0_excl_code_l005_b1000.csv")
    records = []
    for cell, loop, label, family in CELL_SPECS:
        row = _row(corrected_code if family == "code" else primary, loop=loop)
        records.append(
            (
                cell,
                label,
                family,
                max(0.0, _f(row, "policy_gap")),
                max(0.0, _f(row, "info_gap")),
                _f(row, "total_gap"),
            )
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ys = np.arange(len(records))[::-1]
    for y, (_, label, family, recoverable, information, total) in zip(ys, records):
        ax.barh(y, recoverable, color=COLOR[family], height=0.56, label=None)
        ax.barh(y, information, left=recoverable, color=LIGHT, height=0.56, label=None)
        ax.text(total + 0.003, y, f"{total:.3f}", va="center", fontsize=8, color=MUTED)
        if recoverable >= 0.002:
            ax.text(
                recoverable + 0.001,
                y + 0.18,
                f"rec. {recoverable:.3f}",
                ha="left",
                va="bottom",
                fontsize=7,
                color=COLOR[family],
            )

    ax.set_yticks(ys)
    ax.set_yticklabels([record[1] for record in records])
    ax.set_xlabel(r"regret (utility units, $\lambda=0.005$)")
    ax.set_title("Most stopping regret is not recovered by tested visible-signal policies", loc="left", fontsize=11)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=COLOR["writing"], label="recoverable"),
            plt.Rectangle((0, 0), 1, 1, color=LIGHT, label="class-conditional residual"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    ax.set_xlim(0, max(record[5] for record in records) * 1.18)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_decomposition")


# Figure 2: signal-information manipulation within writing-small.
def fig2_mi_recoverable(out_dir: Path) -> None:
    specs = [
        ("8B absolute", "oof_signal_critic8_b1000.csv", "oof_signal_critic8_b1000_bootstrap.csv", "o"),
        ("32B absolute", "oof_signal_critic32_b1000.csv", "oof_signal_critic32_b1000_bootstrap.csv", "s"),
        ("72B absolute", "oof_signal_critic72_b1000.csv", "oof_signal_critic72_b1000_bootstrap.csv", "^"),
        ("32B pairwise", "oof_signal_pairwise32_b1000.csv", "oof_signal_pairwise32_b1000_bootstrap.csv", "D"),
    ]
    points = []
    for label, point_file, boot_file, marker in specs:
        point_row = _read(point_file)[0]
        boot_row = _read(boot_file)[0]
        points.append(
            (
                label,
                _f(point_row, "mi_bits"),
                _f(boot_row, "point"),
                _f(boot_row, "lower95"),
                _f(boot_row, "upper95"),
                marker,
            )
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    absolute = points[:3]
    ax.plot(
        [point[1] for point in absolute],
        [point[2] for point in absolute],
        color=COLOR["writing"],
        alpha=0.4,
        lw=1.2,
        ls="--",
        zorder=1,
    )
    offsets = {
        "8B absolute": (-30, 9),
        "32B absolute": (-10, 12),
        "72B absolute": (8, -15),
        "32B pairwise": (8, 8),
    }
    for label, mi, recoverable, lower, upper, marker in points:
        ax.errorbar(
            [mi],
            [recoverable],
            yerr=_asymmetric_error(recoverable, lower, upper),
            fmt=marker,
            markersize=6,
            color=COLOR["writing"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=3,
            lw=1.3,
            zorder=3,
        )
        dx, dy = offsets[label]
        ax.annotate(label, (mi, recoverable), xytext=(dx, dy), textcoords="offset points", fontsize=8)

    ax.axhline(0.05, color=COLOR["writing"], lw=1.0, ls=":", label=r"writing margin $\Delta=0.05$")
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("mutual information with the continue decision (bits)")
    ax.set_ylabel("reported recoverable gain")
    ax.set_title("More signal information does not monotonically increase decision value", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(bottom=-0.004)
    ax.margins(x=0.14)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_mi_recoverable")


# Figure 3: primary and cross-loop forest plots.
def fig3_forest(out_dir: Path) -> None:
    primary_boot = _read("oof_b1000_l005_bootstrap.csv")
    corrected_code_boot = _read_p0("p0_excl_code_l005_b1000_bootstrap.csv")
    cross_loop = _read("xloop_voi_oof_b1000_l005.csv")

    rows_3a: list[tuple[str, float, float | None, float | None, str]] = []
    code8 = _row(corrected_code_boot, loop="m2_code_8b")
    rows_3a.append(
        (
            "code 8B",
            _f(code8, "recoverable"),
            _f(code8, "recoverable_lower95"),
            _f(code8, "recoverable_upper95"),
            COLOR["code"],
        )
    )
    code32 = _row(corrected_code_boot, loop="m2_code_32b")
    rows_3a.append(
        (
            "code 32B",
            _f(code32, "recoverable"),
            _f(code32, "recoverable_lower95"),
            _f(code32, "recoverable_upper95"),
            COLOR["code"],
        )
    )
    for loop, label, family in (
        ("m2_writing_small_judged", "writing small", "writing"),
        ("m2_writing_mid_judged", "writing mid", "writing"),
        ("qa_m2", "QA", "QA"),
    ):
        row = _row(primary_boot, loop=loop)
        rows_3a.append((label, _f(row, "point"), None, _f(row, "upper95"), COLOR[family]))

    loop_labels = {"code_repair": "code", "writer_critic": "writing", "react_qa": "QA"}
    loop_family = {"code_repair": "code", "writer_critic": "writing", "react_qa": "QA"}
    for row in cross_loop:
        label = loop_labels[row["loop"]]
        color = COLOR[loop_family[row["loop"]]]
        rows_3a.append(
            (
                f"{label} oracle-tau",
                _f(row, "rec_oracle"),
                _f(row, "rec_oracle_lo95"),
                _f(row, "rec_oracle_up95"),
                color,
            )
        )
        rows_3a.append(
            (
                f"{label} transfer-tau",
                _f(row, "rec_transfer"),
                _f(row, "rec_transfer_lo95"),
                _f(row, "rec_transfer_up95"),
                color,
            )
        )

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ys = np.arange(len(rows_3a))[::-1]
    for y, (label, point, lower, upper, color) in zip(ys, rows_3a):
        _forest_row(ax, y, point, lower, upper, color)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.axvline(0.02, color=COLOR["QA"], lw=0.9, ls=":", label=r"code/QA $\Delta=0.02$")
    ax.axvline(0.05, color=COLOR["writing"], lw=0.9, ls=":", label=r"writing $\Delta=0.05$")
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows_3a], fontsize=8)
    ax.set_xlabel("utility increment over OOF heuristic")
    ax.set_title("Reported within-loop gains and cross-loop raw increments", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    values = [value for _, point, lower, upper, _ in rows_3a for value in (point, lower, upper) if value is not None]
    values.extend([0.02, 0.05])
    span = max(values) - min(values)
    ax.set_xlim(min(values) - 0.08 * span, max(values) + 0.08 * span)
    fig.tight_layout()
    _save(fig, out_dir, "fig3a_forest_recoverable")

    rows_3b: list[tuple[str, float, float | None, float | None, str]] = [
        ("code 8B", _f(code8, "point"), _f(code8, "lower95"), _f(code8, "upper95"), COLOR["code"]),
        (
            "code 32B",
            _f(code32, "point"),
            _f(code32, "lower95"),
            _f(code32, "upper95"),
            COLOR["code"],
        ),
    ]
    for loop, label, family in (
        ("m2_writing_small_judged", "writing small", "writing"),
        ("m2_writing_mid_judged", "writing mid", "writing"),
        ("qa_m2", "QA", "QA"),
    ):
        row = _row(primary_boot, loop=loop)
        rows_3b.append(
            (
                label,
                _f(row, "total_regret_point"),
                _f(row, "total_regret_lower95"),
                _f(row, "total_regret_upper95"),
                COLOR[family],
            )
        )

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ys = np.arange(len(rows_3b))[::-1]
    for y, (label, point, lower, upper, color) in zip(ys, rows_3b):
        _forest_row(ax, y, point, lower, upper, color)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows_3b], fontsize=8)
    ax.set_xlabel("total stopping regret (utility units)")
    ax.set_title("Total regret remains largest in critic-driven writing", loc="left", fontsize=10)
    ax.set_xlim(left=-0.003)
    fig.tight_layout()
    _save(fig, out_dir, "fig3b_forest_total_regret")


# Figure 4: predictive accuracy versus deployed utility.
def fig4_auroc_dissociation(out_dir: Path) -> None:
    rows = _read("xloop_voi_oof_b1000_l005.csv")
    order = ["code_repair", "writer_critic", "react_qa"]
    rows = sorted(rows, key=lambda row: order.index(row["loop"]))
    family = {"code_repair": "code", "writer_critic": "writing", "react_qa": "QA"}
    labels = {"code_repair": "code", "writer_critic": "writing", "react_qa": "QA"}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.arange(len(rows))
    colors = [COLOR[family[row["loop"]]] for row in rows]

    ax = axes[0]
    aurocs = [_f(row, "auroc_test") for row in rows]
    bars = ax.bar(x, aurocs, color=colors, width=0.55)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[row["loop"]] for row in rows])
    ax.set_ylabel("held-out AUROC")
    ax.set_title("predictive discrimination", loc="left", fontsize=10)
    for bar, value in zip(bars, aurocs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[1]
    width = 0.34
    oracle = [_f(row, "rec_oracle") for row in rows]
    transfer = [_f(row, "rec_transfer") for row in rows]
    ax.bar(x - width / 2, oracle, width=width, color=colors, alpha=0.55, label="oracle-tau")
    ax.bar(x + width / 2, transfer, width=width, color=colors, label="transfer-tau")
    for index, row in enumerate(rows):
        ax.errorbar(
            [x[index] - width / 2],
            [oracle[index]],
            yerr=_asymmetric_error(oracle[index], _f(row, "rec_oracle_lo95"), _f(row, "rec_oracle_up95")),
            fmt="none",
            ecolor=INK,
            capsize=2,
            lw=1.0,
        )
        ax.errorbar(
            [x[index] + width / 2],
            [transfer[index]],
            yerr=_asymmetric_error(transfer[index], _f(row, "rec_transfer_lo95"), _f(row, "rec_transfer_up95")),
            fmt="none",
            ecolor=INK,
            capsize=2,
            lw=1.0,
        )
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[row["loop"]] for row in rows])
    ax.set_ylabel("utility increment over OOF heuristic")
    ax.set_title("deployed decision value", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle(r"Predictive accuracy does not imply deployed value ($\lambda=0.005$)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, out_dir, "fig4_auroc_dissociation")


# Figure 5: P0-corrected descriptive same-task capability analysis.
def fig5_f9_capability(out_dir: Path) -> None:
    rows = _read("capability_difficulty_p0.csv")
    tier_values = {
        (row["tier"], row["model"]): float(row["value"])
        for row in rows
        if row["section"] == "tier" and row["metric"] == "mean_regret"
    }
    paired = next(row for row in rows if row["metric"] == "paired_mean_difference")
    paired_t = next(row for row in rows if row["metric"] == "paired_t_analogue")
    scale_coef = next(
        row
        for row in rows
        if row["metric"] == "standardized_coefficient" and row["model"] == "model_size"
    )
    difficulty_coef = next(
        row
        for row in rows
        if row["metric"] == "standardized_coefficient" and row["model"] == "difficulty"
    )
    tiers = ["easy", "medium", "hard"]
    x = np.arange(len(tiers))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(
        x - width / 2,
        [tier_values[(tier, "gen_32b")] for tier in tiers],
        width=width,
        color=COLOR["code"],
        alpha=0.5,
        label="32B",
    )
    ax.bar(
        x + width / 2,
        [tier_values[(tier, "gen_8b")] for tier in tiers],
        width=width,
        color=COLOR["code"],
        label="8B",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([tier.capitalize() for tier in tiers])
    ax.set_ylabel(r"mean oracle regret ($\lambda=0.005$)")
    ax.set_title("Stopping regret tracks model capability, not task difficulty", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.98,
        0.95,
        f"paired 8B - 32B = {float(paired['value']):+.3f} "
        f"(n={paired['n']}, t~{float(paired_t['value']):.2f})\n"
        f"standardized coefficients: model {float(scale_coef['value']):+.3f}; "
        f"difficulty {float(difficulty_coef['value']):+.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "square,pad=0.3", "facecolor": "white", "edgecolor": GRID},
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig5_f9_capability")


# Figure 6: signal-only model-class ladder and truth-fed positive controls.
LADDER_ORDER = ["gbt", "gru-h8", "gru-h16", "gru-h32", "gru-h64", "gru-h128", "lstm-h64"]
LADDER_LABELS = ["GBT", "h8", "h16", "h32", "h64", "h128", "LSTM"]
MC3_FILES = {
    "code_8b": "p0_excl_mc3_m2_code_8b.csv",
    "code_32b": "p0_excl_mc3_m2_code_32b.csv",
    "writing_small": "mc3_m2_writing_small_judged.csv",
    "writing_mid": "mc3_m2_writing_mid_judged.csv",
    "QA": "mc3_qa_m2.csv",
}


def fig6_gru_ladder(out_dir: Path) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.9),
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    x = np.arange(len(LADDER_ORDER))
    styles = {
        "code_8b": ("o", "-"),
        "code_32b": ("s", "--"),
        "writing_small": ("o", "-"),
        "writing_mid": ("s", "--"),
        "QA": ("D", "-"),
    }
    cached: dict[str, dict[str, dict[str, str]]] = {}
    for cell in CELLS:
        read_rows = _read_p0(MC3_FILES[cell]) if FAMILY[cell] == "code" else _read(MC3_FILES[cell])
        cached[cell] = {row["member"]: row for row in read_rows}
        rows = cached[cell]
        values = np.array([_f(rows[member], "recoverable") for member in LADDER_ORDER])
        lower = np.array([_f(rows[member], "lo95") for member in LADDER_ORDER])
        upper = np.array([_f(rows[member], "up95") for member in LADDER_ORDER])
        marker, linestyle = styles[cell]
        ax_top.errorbar(
            x,
            values,
            yerr=np.vstack([np.maximum(0, values - lower), np.maximum(0, upper - values)]),
            marker=marker,
            markersize=4,
            color=COLOR[FAMILY[cell]],
            lw=1.3,
            ls=linestyle,
            capsize=2,
            label=CELL_LABEL[cell],
            alpha=0.9,
        )

    ax_top.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax_top.axhline(0.02, color=COLOR["QA"], lw=0.9, ls=":", label=r"code/QA $\Delta=0.02$")
    ax_top.axhline(0.05, color=COLOR["writing"], lw=0.9, ls=":", label=r"writing $\Delta=0.05$")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(LADDER_LABELS)
    ax_top.set_ylabel("raw utility increment")
    ax_top.set_title("Signal-only model-class ladder", loc="left", fontsize=10)
    ax_top.legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    control_member = "gru-h64+truth(bidi)"
    x2 = np.arange(len(CELLS))
    values = np.array([_f(cached[cell][control_member], "recoverable") for cell in CELLS])
    lower = np.array([_f(cached[cell][control_member], "lo95") for cell in CELLS])
    upper = np.array([_f(cached[cell][control_member], "up95") for cell in CELLS])
    colors = [COLOR[FAMILY[cell]] for cell in CELLS]
    ax_bottom.bar(x2, values, color=colors, width=0.56)
    ax_bottom.errorbar(
        x2,
        values,
        yerr=np.vstack([np.maximum(0, values - lower), np.maximum(0, upper - values)]),
        fmt="none",
        ecolor=INK,
        capsize=2,
        lw=1.0,
    )
    ax_bottom.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax_bottom.set_xticks(x2)
    ax_bottom.set_xticklabels([CELL_LABEL[cell] for cell in CELLS], fontsize=8)
    ax_bottom.set_ylabel("raw utility increment")
    ax_bottom.set_title("Truth-fed bidirectional GRU positive control", loc="left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.86, 1])
    _save(fig, out_dir, "fig6_gru_ladder")


def graphical_abstract(out_dir: Path) -> None:
    """Generate a submission-scale overview without implying a five-cell null."""
    fig = plt.figure(figsize=(13.28, 5.31), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.92,
        "When should an iterative LLM loop stop?",
        fontsize=22,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.035,
        0.84,
        "Measure the utility available to signal-only stopping policies before building another gate.",
        fontsize=11.5,
        color=MUTED,
        va="top",
    )

    for x in (0.335, 0.675):
        ax.plot([x, x], [0.10, 0.77], color=GRID, lw=1.1)

    ax.text(0.035, 0.745, "1  Collect full-horizon loops", fontsize=13, fontweight="bold")
    row_specs = [
        ("Code", "repair", "visible tests", COLOR["code"], [0.30, 0.55, 0.78, 0.68, 0.88]),
        ("Writing", "revise", "critic score", COLOR["writing"], [0.45, 0.58, 0.52, 0.67, 0.62]),
        ("QA", "retrieve", "confidence", COLOR["QA"], [0.24, 0.44, 0.39, 0.61, 0.55]),
    ]
    xs = np.linspace(0.135, 0.305, 5)
    for idx, (family, action, signal, color, values) in enumerate(row_specs):
        y0 = 0.62 - idx * 0.18
        ys = y0 + (np.asarray(values) - 0.5) * 0.105
        ax.text(0.04, y0 + 0.025, family, color=color, fontsize=12, fontweight="bold", va="center")
        ax.text(0.04, y0 - 0.028, f"{action}  |  {signal}", color=MUTED, fontsize=8.5, va="center")
        ax.plot(xs, ys, color=color, lw=2.3)
        ax.scatter(xs, ys, s=26, facecolor="white", edgecolor=color, linewidth=1.6, zorder=3)
        ax.plot(xs, y0 - 0.045 + 0.018 * np.sin(np.arange(5)), color=MUTED, lw=1.0, ls="--")
    ax.text(0.135, 0.115, "round 1", fontsize=8.5, color=MUTED, ha="center")
    ax.text(0.305, 0.115, "round T", fontsize=8.5, color=MUTED, ha="center")
    ax.text(0.22, 0.075, "hidden quality retained for offline replay", fontsize=9, ha="center")

    ax.text(0.37, 0.745, "2  Decompose stopping regret", fontsize=13, fontweight="bold")
    axis_y = 0.50
    h_x, v_x, o_x = 0.39, 0.455, 0.64
    ax.plot([h_x, o_x], [axis_y, axis_y], color=INK, lw=1.3)
    ax.scatter([h_x, v_x, o_x], [axis_y] * 3, s=[50, 50, 62], color=[MUTED, COLOR["writing"], INK], zorder=3)
    ax.text(h_x, axis_y + 0.07, "OOF\nheuristic", fontsize=9, ha="center", va="bottom")
    ax.text(v_x, axis_y - 0.075, "best tested\nvisible policy", fontsize=9, ha="center", va="top")
    ax.text(o_x, axis_y + 0.07, "hindsight\noracle", fontsize=9, ha="center", va="bottom")
    ax.annotate("", xy=(v_x, 0.34), xytext=(h_x, 0.34), arrowprops=dict(arrowstyle="|-|", color=COLOR["writing"], lw=2))
    ax.annotate("", xy=(o_x, 0.34), xytext=(v_x, 0.34), arrowprops=dict(arrowstyle="|-|", color=LIGHT, lw=5))
    ax.text((h_x + v_x) / 2, 0.285, "reported\nrecoverable gain", color=COLOR["writing"], fontsize=9, ha="center", va="top")
    ax.text((v_x + o_x) / 2, 0.285, "class-conditional\nresidual", color=MUTED, fontsize=9, ha="center", va="top")
    ax.text(0.505, 0.125, "Task-clustered BCa endpoints  |  margins fixed by family", fontsize=9.5, ha="center")

    ax.text(0.71, 0.745, "3  Test practical materiality", fontsize=13, fontweight="bold")
    ax.text(0.715, 0.64, "4 / 5", fontsize=43, fontweight="bold", color=INK, va="center")
    ax.text(
        0.845,
        0.65,
        "primary family-specific\ntargets exclude material gain",
        fontsize=10.0,
        va="center",
        linespacing=1.35,
    )
    ax.plot([0.715, 0.955], [0.485, 0.485], color=GRID, lw=1.0)
    ax.text(0.715, 0.425, "Code-8B total regret", fontsize=11.5, fontweight="bold", color=COLOR["code"])
    ax.text(0.715, 0.365, "remains inconclusive at the 0.02 margin", fontsize=10.5, color=MUTED)
    ax.text(0.715, 0.245, "Predictive accuracy does not guarantee\nstopping utility.", fontsize=11.0, fontweight="bold", linespacing=1.3)
    ax.text(0.715, 0.135, r"Primary: $\lambda=0.005$  |  generation-only cost  |  stop-at-last", fontsize=8.5, color=MUTED)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cas-grabs.pdf", facecolor="white")
    # A tiny DPI offset avoids floating-point truncation to 1592 px in Matplotlib.
    fig.savefig(out_dir / "cas-grabs.png", facecolor="white", dpi=300.01)
    plt.close(fig)
    print(f"  wrote {out_dir / 'cas-grabs'}.{{pdf,png}}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--p0-results-dir", default=str(DEFAULT_P0_RESULTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_FIGURES))
    parser.add_argument("--only", nargs="*", choices=["1", "2", "3", "4", "5", "6", "g"])
    args = parser.parse_args()

    global RESULTS, P0_RESULTS
    RESULTS = Path(args.results_dir).resolve()
    P0_RESULTS = Path(args.p0_results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    functions = {
        "1": fig1_decomposition,
        "2": fig2_mi_recoverable,
        "3": fig3_forest,
        "4": fig4_auroc_dissociation,
        "5": fig5_f9_capability,
        "6": fig6_gru_ladder,
        "g": graphical_abstract,
    }
    # Figure 5 is a descriptive analysis and remains opt-in unless the
    # manuscript explicitly includes it.
    selected = args.only or ["1", "2", "3", "4", "6", "g"]
    print(f"result directory: {RESULTS}")
    print(f"corrected P0 code results: {P0_RESULTS}")
    for key in selected:
        print(f"[fig {key}] {functions[key].__name__}")
        functions[key](out_dir)


if __name__ == "__main__":
    main()
