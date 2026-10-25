"""Learning-curve figures (matplotlib), rendered once for a light and once for a dark background.

Colours follow a validated categorical order (blue, orange, aqua, yellow - adjacent pairs CVD-safe in
both modes, at most four series, never cycled);
reference lines for the baselines are drawn in muted ink and labelled directly. The tables
in results/*.md are the text equivalent of every figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}


def learning_curves(curves: Dict[str, Dict[str, Optional[dict]]], references: Dict[str, Dict[str, float]],
                    labels: Dict[str, str], out: Path, theme: str = "light", colors: Optional[Dict[str, int]] = None) -> Path:
    """One panel per scenario: validation return vs environment steps (mean ± std over seeds).

    curves[scenario][algo] = {"steps", "mean", "std", "n_seeds"}; references[scenario][name] = return.
    `colors` fixes each algo's series colour across panels (default: its position in the panel); the
    legend lists every series once."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    scenarios = list(curves)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.2 * len(scenarios), 4.2), squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    for ax, scenario in zip(axes[0], scenarios):
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"]); ax.spines[side].set_linewidth(1)
        ax.grid(axis="y", color=t["grid"], linewidth=1, linestyle="-"); ax.set_axisbelow(True)
        ax.tick_params(colors=t["muted"], labelsize=9, length=0)
        x_max = 0
        for i, (algo, c) in enumerate(curves[scenario].items()):
            if not c:
                continue
            x = [s / 1e6 for s in c["steps"]]
            lo = [m - s for m, s in zip(c["mean"], c["std"])]
            hi = [m + s for m, s in zip(c["mean"], c["std"])]
            color = t["series"][(colors or {}).get(algo, i)]
            ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
            ax.plot(x, c["mean"], color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    label=f"{labels.get(algo, algo)} (n={c['n_seeds']})")
            ax.plot(x[-1:], c["mean"][-1:], "o", color=color, markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
            x_max = max(x_max, x[-1])
        refs = sorted(references.get(scenario, {}).items(), key=lambda kv: -kv[1])
        lo_y, hi_y = ax.get_ylim()
        lo_y, hi_y = min([lo_y] + [v for _, v in refs]) - 10, max([hi_y] + [v for _, v in refs]) + 10
        ax.set_ylim(lo_y, hi_y)
        prev = hi_y                       # the top of the axes counts as the line above the first reference
        for name, value in refs:          # label above its line, or below it when the line above is too close
            below = prev - value < 0.06 * (hi_y - lo_y)
            ax.axhline(value, color=t["muted"], linewidth=1)
            ax.annotate(f"{name} {value:.0f}", xy=(0, value), xycoords=("axes fraction", "data"),
                        xytext=(6, -3 if below else 3), textcoords="offset points", fontsize=8.5, color=t["ink2"],
                        ha="left", va="top" if below else "bottom", zorder=5,
                        bbox={"boxstyle": "square,pad=0.15", "facecolor": t["surface"], "edgecolor": "none", "alpha": 0.85})
            prev = value
        ax.set_title(scenario.replace("_", " "), color=t["ink"], fontsize=11, loc="left", pad=8)
        ax.set_xlabel("environment steps (millions)", color=t["ink2"], fontsize=9)
        ax.set_ylabel("validation return (mean over seeds)", color=t["ink2"], fontsize=9)
        ax.set_xlim(0, x_max * 1.02 if x_max else 1)
    handles, names = [], []                         # every series once, in order of appearance
    for ax in axes[0]:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h); names.append(n)
    if handles:
        leg = fig.legend(handles, names, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
                         bbox_to_anchor=(0.5, -0.01))
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    return out
