"""Draw the paper's quantitative displays from the computed frames."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from mode_plots import changes_figure, totals_figure

STATES = ["Clean win", "Completion gain", "Mirage success", "Process cleanup", "Status quo",
          "Hidden deterioration", "Compensating decline", "Completion loss", "Joint deterioration"]
SHORT = ["C", "G", "M", "P", "S", "H", "D", "L", "J"]
COLORS = ["#8fcaa1","#dcebdc","#e99a82","#b6dbbf","#efede5","#f2ba95","#dedced","#c6d5e4","#bfafa6"]
ARMS = ["G1", "G2", "G3", "G4", "G5"]
STATE_CMAP = ListedColormap(COLORS)


def overall(table):
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.1), layout="constrained")
    for ax, metric, scale, title in zip(axes, ["completion", "flags"], [100, 1],
                                        ["Completion (percentage points)", "Distinct failure modes per run"]):
        frame = table[table.metric == metric].set_index("group").loc[["Pooled", *ARMS]][::-1]
        effects = frame.estimate * scale
        err = [(frame.estimate - frame.ci_low) * scale, (frame.ci_high - frame.estimate) * scale]
        ax.errorbar(effects, np.arange(len(frame)), xerr=err, fmt="o", color="#23694e", capsize=3)
        ax.set(yticks=range(len(frame)), yticklabels=frame.index, title=title, xlabel="Specialist minus generalist")
        ax.axvline(0,color=".55",ls=":",lw=1)
    return fig


def state_panels(counts):
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.7), layout="constrained")
    for group, ax in zip([*ARMS, "Pooled"], axes.flat):
        totals = counts.set_index("group").loc[group,STATES].to_numpy(dtype=int).reshape(3,3)
        ax.imshow(np.arange(9).reshape(3,3),cmap=STATE_CMAP,vmin=0,vmax=8)
        for i,j in np.ndindex(3,3):
            ax.text(j,i,f"{totals[i,j]}\n{totals[i,j]/totals.sum():.0%}",va="center",ha="center",fontsize=9)
        ax.set(xticks=range(3), xticklabels=["Fewer","Same","More"], yticks=range(3),
               yticklabels=["Higher","Same","Lower"], xlabel="Flagged modes",ylabel="Completion",title=group)
    return fig


def frequency(table):
    rows = table.sort_values(["risk_frequency","display_task"], ascending=[False,True])
    values = rows[ARMS].replace(dict(zip(STATES,range(9)))).to_numpy(dtype=int)
    fig,(ax,bars)=plt.subplots(1,2,figsize=(8,8.8),width_ratios=(1.2,1),layout="constrained")
    ax.imshow(values,cmap=STATE_CMAP,aspect="auto",vmin=0,vmax=8)
    for row,col in np.ndindex(values.shape):
        ax.text(col,row,SHORT[values[row,col]],va="center",ha="center",fontsize=7)
    ax.set(yticks=range(len(rows)),yticklabels=rows.display_task,xticks=range(5),xticklabels=ARMS,title="Case state")
    ax.tick_params(labelsize=8)
    bars.barh(np.arange(len(rows)),100*rows.risk_frequency,color="#c78351")
    bars.set(ylim=ax.get_ylim(),xlim=(0,100),yticks=[],xlabel="Groups in a risk state (%)",title="Risk frequency")
    fig.text(.5,-.015,"M = Mirage success; H = Hidden deterioration. See state_counts.csv for all states.",ha="center",fontsize=8)
    return fig


def heatmaps(data, ordering, dimension, names, only_difference):
    groups = ["Risk minus Benefit"] if only_difference else ["Risk","Benefit"]
    modes = list(dict.fromkeys(ordering["mode"]))
    if dimension == "role":
        labels = ["CEO","CPO","CTO","Coder","Reviewer"]
    else:
        labels = list(dict.fromkeys(ordering[~ordering.label.isin(["Other issues","Focal mode total"])].label)) + ["Other issues", "Focal mode total"]
    fig,axes = plt.subplots(1,len(groups),squeeze=False, sharey=True,
                            figsize=(4.9*len(groups) + (2 if dimension == "label" else 0),
                                     max(3.1, .25*len(labels)+1.3)),layout="constrained")
    limit = max(1,data.raw_change.abs().max())
    for ax,category in zip(axes.flat,groups):
        frame=data[data.category==category].pivot(index=dimension,columns="mode",values="raw_change").reindex(index=labels,columns=modes)
        values=frame.to_numpy()
        ax.imshow(values,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect="auto")
        for i,j in zip(*np.where(np.isfinite(values))):
            n=values[i,j]
            ax.text(j,i,f"{n:+.0f}" if n else "0",ha="center",va="center",fontsize=7.3,
                    color="white" if abs(n)>limit*.55 else "#252525")
        ax.set(xticks=range(len(modes)),xticklabels=modes,yticks=range(len(labels)),
               yticklabels=[names.get(x,x) for x in labels],title=category,xlabel="MAST failure mode")
        ax.tick_params(labelsize=8)
    return fig


def issue_panels(cells, outside, names):
    """Compare changes in checked issues on a common color scale and row order."""
    modes = cells["mode"].drop_duplicates().tolist()
    summaries = ["Other issues", "Focal mode total"]
    named = cells.loc[~cells.label.isin(summaries), "label"].drop_duplicates().tolist()
    rows = named + summaries
    raw_limit = max(1, cells.loc[cells.label.ne("Focal mode total"), "raw_change"].abs().max())
    norm = TwoSlopeNorm(vmin=-raw_limit, vcenter=0, vmax=raw_limit)
    # Reserve separate widths for the long issue names, the two matrices, and their gutter.
    fig = plt.figure(figsize=(6.8, 4.55))
    grid = fig.add_gridspec(1, 4, width_ratios=(2.85, 1.73, .43, 1.73),
                            left=.012, right=.988, bottom=.065, top=.885, wspace=0)
    labels_ax = fig.add_subplot(grid[0, 0])
    risk_ax = fig.add_subplot(grid[0, 1], sharey=labels_ax)
    benefit_ax = fig.add_subplot(grid[0, 3], sharey=labels_ax)
    labels_ax.set_axis_off()
    labels_ax.set_xlim(0, 1)
    labels_ax.set_ylim(len(rows)-.5, -.5)
    for i, label in enumerate(rows):
        labels_ax.text(0, i, names.get(label, label), va="center", fontsize=6.9,
                       weight="bold" if label == "Focal mode total" else "normal", color="#20292c")
    labels_ax.set_title("Checked issue type", loc="left", fontsize=8.1, weight="bold", pad=4)

    # Alternating row bands support comparisons across the gutter; rules separate mode blocks.
    label_mode = cells.loc[cells.label.isin(named)].drop_duplicates("label").set_index("label")["mode"]
    boundaries = [i-.5 for i in range(1, len(named)) if label_mode[rows[i]] != label_mode[rows[i-1]]]
    for ax in (labels_ax, risk_ax, benefit_ax):
        for i in range(1, len(named), 2):
            ax.axhspan(i-.5, i+.5, color="#f5f6f6", lw=0, zorder=-1)
        for boundary in boundaries:
            ax.axhline(boundary, color="#d8ddde", lw=.38, zorder=1)
        ax.axhline(len(named)-.5, color="#9ca6a8", lw=.55)
        ax.axhline(len(rows)-1.5, color="#9ca6a8", lw=.55)

    outside = outside.set_index("category").raw_change
    for ax, category in zip((risk_ax, benefit_ax), ("Risk", "Benefit")):
        panel = cells[cells.category.eq(category)].pivot(index="label", columns="mode", values="raw_change")
        values = panel.reindex(index=rows, columns=modes)
        ax.imshow(values.iloc[:-1], cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        # Highlight within each panel because a mode's two largest changes can differ by case state.
        leaders = set()
        for mode in modes:
            magnitudes = values.loc[named, mode].abs().dropna()
            for label in magnitudes[magnitudes > 0].nlargest(2).index:
                leaders.add((label, mode))
        for i, label in enumerate(rows):
            for j, mode in enumerate(modes):
                value = values.at[label, mode]
                if not np.isfinite(value):
                    continue
                is_total = label == "Focal mode total"
                if not is_total:
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor="#a2a2a2", lw=.35))
                ax.text(j, i, f"{value:+.0f}" if value else "0", ha="center", va="center", fontsize=7.0,
                        color="white" if not is_total and abs(value) > .65*raw_limit else "#222222",
                        weight="bold" if is_total or (label, mode) in leaders else "normal")
        ax.spines[:].set_visible(False)
        ax.set_ylim(len(rows)-.5, -.5)
        ax.set_xticks(range(len(modes)), labels=modes, fontsize=7)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(f"{category} cases", fontsize=9, weight="bold", pad=6)
        ax.set_xlabel(f"Outside focal modes: {outside[category]:+d}", fontsize=6.9, labelpad=4)
    return fig


def feature_plot(models):
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    limit = max(1, 120 * models[["ci_low", "ci_high"]].abs().to_numpy().max())
    for index, row in enumerate(models.itertuples()):
        significant = row.p_raw < .05
        color = "#C46210" if significant else "#222222"
        estimate = 100 * row.effect
        ax.errorbar(estimate, index,
                    xerr=[[estimate - 100 * row.ci_low], [100 * row.ci_high - estimate]],
                    fmt="o", color=color, capsize=3)
        ax.annotate(f"{estimate:+.1f}", (estimate, index), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10, color=color,
                    weight="bold" if significant else "normal")
    ax.axvline(0, color="#777", lw=.8)
    ax.set(xlim=(-limit, limit), ylim=(len(models) - .45, -.55), yticks=range(len(models)),
           yticklabels=models.dimension,
           xlabel="Change in risk frequency (percentage points per unit feature score)")
    ax.tick_params(axis="y", labelsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def render(tables,out):
    folder=out/"figures"
    folder.mkdir(exist_ok=True)
    # Earlier releases wrote the pooled sixth panel a second time on its own.
    (folder / "pooled_all_split.pdf").unlink(missing_ok=True)
    def save(figure,name):
        figure.savefig(folder/f"{name}.pdf",bbox_inches="tight",metadata={"CreationDate": None})
        plt.close(figure)
    save(overall(tables["overall"]),"outcome_effects")
    save(state_panels(tables["state_counts"]),"case_states_by_group")
    estimates=tables["mode_contrasts"]
    modes=sorted(estimates["mode"].unique())
    save(totals_figure(estimates,modes,["All"]),"failure_mode_counts")
    save(frequency(tables["task_risk_frequency"]),"task_risk_frequency")
    save(changes_figure(estimates,modes,by_category=True),"failure_mode_contrasts")
    names=tables["issue_inventory"].set_index("label").issue_type.to_dict()
    save(issue_panels(tables["issue_cells"], tables["outside_focal_modes"], names), "issue_changes_by_state")
    for table,filename in [("issue_difference", "issue_change_contrasts"),
                           ("role_cells", "role_changes_by_state"),
                           ("role_difference", "role_change_contrasts")]:
        ordering = tables["role_cells"] if table.startswith("role") else tables["issue_cells"]
        save(heatmaps(tables[table],ordering,"role" if table.startswith("role") else "label",names,
                      table.endswith("difference")),filename)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42}):
        save(feature_plot(tables["feature_models"]), "task_feature_associations")
