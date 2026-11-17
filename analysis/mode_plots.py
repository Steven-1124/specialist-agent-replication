"""Mode totals and paired-case contrast graphics from the experimental report."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

GENERALIST, SPECIALIST = "#28734B", "#A63A32"
RISK, BENEFIT = "#A63A32", "#28734B"
DIFFERENCE = "Risk minus Benefit"
ORANGE = "#D47822"


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def totals_figure(results, modes, contrasts, *, height=3.0):
    by_category = len(contrasts) > 1
    offset = .22 if by_category else .20
    fig, axes = plt.subplots(
        1, len(contrasts), figsize=(12.8, 4.7) if by_category else (8.4, height),
        sharey=True, layout=None if by_category else "constrained", squeeze=False,
    )
    x = np.arange(len(modes))
    for ax, contrast in zip(axes[0], contrasts):
        rows = results[results.contrast.eq(contrast)].set_index("mode").loc[modes]
        for column, shift, color in [
            ("generalist", -offset, GENERALIST), ("specialist", offset, SPECIALIST),
        ]:
            bars = ax.bar(x + shift, rows[column], width=.38, color=color, label=column.title())
            labels = ax.bar_label(bars, padding=2.5, fontsize=9.6 if by_category else 8.5)
            for label in labels:
                label.set_x(np.copysign(1, shift))
        ax.set(xticks=x, xticklabels=modes, xlabel="Failure mode")
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=.19)
        if by_category:
            ax.set_title(f"{contrast} ({int(rows.n.iloc[0])} cases)", fontsize=15, weight="bold")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    left_ax = axes[0, 0]
    left_ax.set_ylabel("Total occurrences")
    if by_category:
        fig.subplots_adjust(left=.07, right=.988, bottom=.26, top=.88, wspace=.15)
        fig.legend(*left_ax.get_legend_handles_labels(), frameon=False, loc="lower center",
                   ncol=2, bbox_to_anchor=(.5, .02))
    else:
        left_ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes[0]:
        for pair in zip(ax.texts[:len(modes)], ax.texts[len(modes):]):
            lower, higher = sorted(pair, key=lambda label: label.xy[1])
            bottom, top = (label.get_window_extent(renderer) for label in (lower, higher))
            if bottom.overlaps(top):
                # Raise the label on the taller bar enough to clear its partner.
                lift = (bottom.y1 - top.y0) * 72 / fig.dpi
                higher.set_y(higher.get_position()[1] + lift + 2)
    return fig


def change_bars(ax, rows, offset=0, color=None, label=None, significant_labels_only=False,
                value_suffix=""):
    x = np.arange(len(rows)) + offset
    significant = rows.p_raw.lt(.05)
    fill = np.where(significant, ORANGE, "#282828") if color is None else color
    heights = rows.estimate.to_numpy()
    ax.bar(x, heights, width=.58 if color is None else .34, color=fill, label=label, zorder=2)
    ax.errorbar(x, heights, [heights - rows.ci_low, rows.ci_high - heights],
                fmt="none", color="#333333", capsize=2.3, lw=.85, zorder=3)
    for position, row in zip(x, rows.itertuples()):
        if significant_labels_only and row.p_raw >= .05:
            continue
        above = row.estimate >= 0
        endpoint = row.ci_high if above else row.ci_low
        value = f"{row.estimate:+.2f}" if abs(row.estimate) >= .005 else "0.00"
        ax.annotate(value + value_suffix + stars(row.p_raw), (position, endpoint),
                    xytext=(0, 4 if above else -4), textcoords="offset points",
                    ha="center", va="bottom" if above else "top",
                    fontsize=9.5, color=ORANGE if row.p_raw < .05 else "#222222",
                    weight="bold" if row.p_raw < .05 else "normal", zorder=4)


def changes_figure(results, modes, by_category=False, *, percentage_points=False):
    if by_category:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.1), sharey=True, layout="constrained",
                                 gridspec_kw={"width_ratios": [1.1, 1]})
        contrasts = ["Risk", "Benefit", DIFFERENCE]
        axes[0].set_title("Generalist → Specialist", fontsize=13, weight="bold")
        axes[1].set_title("Risk − Benefit", fontsize=13, weight="bold")
    else:
        fig, ax = plt.subplots(figsize=(8.4, 3.0), layout="constrained")
        axes = [ax]
        contrasts = ["All"]
    shown = results[results.contrast.isin(contrasts)].copy()
    if percentage_points:
        shown[["estimate", "ci_low", "ci_high"]] *= 100 / 3
    limit = max(.05, shown[["ci_low", "ci_high"]].abs().max().max()) * 1.29
    for contrast in contrasts:
        rows = shown[shown.contrast.eq(contrast)].set_index("mode").loc[modes]
        if contrast in ("Risk", "Benefit"):
            shift, color = (-.18, RISK) if contrast == "Risk" else (.18, BENEFIT)
            change_bars(axes[0], rows, offset=shift, color=color, label=contrast,
                        significant_labels_only=True)
        else:
            change_bars(axes[-1], rows, significant_labels_only=by_category,
                        value_suffix="%" if percentage_points else "")
    x_padding = 1.05 if by_category else .65
    for ax in axes:
        ax.axhline(0, color="#777777", lw=.8, zorder=1)
        ax.set(xlim=(-x_padding, len(modes)-1+x_padding), ylim=(-limit, limit),
               xticks=np.arange(len(modes)), xticklabels=modes, xlabel="Failure mode")
        ax.yaxis.set_major_locator(MaxNLocator(6, symmetric=True, steps=[1, 2, 2.5, 5, 10]))
        if percentage_points:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Change in incidence (pp)" if percentage_points
                       else "Change in occurrences per case")
    if by_category:
        axes[0].legend(frameon=False, loc="upper left", ncol=2)
    return fig
