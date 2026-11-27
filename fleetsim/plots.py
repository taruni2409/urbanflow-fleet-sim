"""Static charts for the README (matplotlib, light theme)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_MUTED = "#6b6a63"
GRID = "#e4e3dd"
SERIES = {"greedy": "#2a78d6", "batched": "#eb6834"}          # categorical slots 1, 2
LABELS = {"greedy": "Greedy (first come, first served)", "batched": "Batched (Hungarian matching)"}
STATE_COLORS = {"serving": "#2a78d6", "idle": "#1baf7a", "charging": "#eda100"}


def _style(ax, title, ylabel, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def fleet_size_panels(summary, path, city_name):
    panels = [
        ("revenue_per_vehicle_hour", "Revenue per vehicle-hour (£)", "£ / vehicle-hour"),
        ("fulfilment", "Share of requests served", "fulfilment"),
        ("p90_wait_min", "90th-percentile rider wait (min)", "minutes"),
        ("empty_km_share", "Share of km driven empty", "empty km / all km"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for ax, (col, title, ylabel) in zip(axes.ravel(), panels):
        for pol, g in summary.groupby("policy"):
            g = g.sort_values("n_vehicles")
            x, y, ci = g["n_vehicles"], g[col], g[f"{col}_ci95"]
            ax.fill_between(x, y - ci, y + ci, color=SERIES[pol], alpha=0.15, linewidth=0)
            ax.plot(x, y, color=SERIES[pol], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[pol])
        _style(ax, title, ylabel, "fleet size (vehicles)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.975), labelcolor=INK)
    n_seeds = int(summary["n_seeds"].iloc[0])
    fig.suptitle(f"{city_name}: fleet size vs performance  (mean of {n_seeds} seeds, band = 95% CI)",
                 x=0.01, ha="left", y=0.995, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def day_timeline(timeline, path, title):
    t = timeline["t_min"] / 60
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    bottom = None
    for state in ("serving", "idle", "charging"):
        y = timeline[state]
        base = 0 if bottom is None else bottom
        ax1.fill_between(t, base, base + y, step="post", color=STATE_COLORS[state],
                         alpha=0.85, linewidth=0, label=state)
        bottom = base + y
    _style(ax1, "What the fleet is doing through the day", "vehicles", "")
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncol=3, labelcolor=INK)

    new_req = timeline["requests_so_far"].diff().fillna(timeline["requests_so_far"])
    ax2.step(t, new_req, where="post", color="#2a78d6", linewidth=2, label="new requests / 15 min")
    ax2.step(t, timeline["waiting"], where="post", color="#eb6834", linewidth=2,
             label="riders waiting (snapshot)")
    _style(ax2, "Demand vs riders left waiting", "riders", "hour of day")
    ax2.legend(loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9, ncol=2, labelcolor=INK)
    ax2.set_xticks(range(0, 25, 3))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
