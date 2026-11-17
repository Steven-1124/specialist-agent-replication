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
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.8, 4.75), layout="constrained")
    fig.get_layout_engine().set(w_pad=.025, h_pad=.025, wspace=.045)
    outside = outside.set_index("category").raw_change
    for ax, category in zip(axes, ("Risk", "Benefit")):
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
        ax.set_xticks(range(len(modes)), labels=modes, fontsize=7.2)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=2)
        ax.set_title(f"{category} cases", fontsize=10, weight="bold", pad=6)
        ax.axhline(len(named)-.5, color="#aaa", lw=.45)
        ax.axhline(len(rows)-1.5, color="#777", lw=.45)
        ax.set_xlabel(f"Outside focal modes: {outside[category]:+d}", fontsize=7.0, labelpad=4)
    axes[0].set_yticks(range(len(rows)), labels=[names.get(label, label) for label in rows], fontsize=7)
    axes[0].get_yticklabels()[-1].set_weight("bold")
    return fig


def feature_plot(models):
    fig,ax = plt.subplots(figsize=(8.4,3.4),layout="constrained")
    for i,row in enumerate(models.itertuples()):
        ax.errorbar(100*row.effect,i,xerr=[[100*(row.effect-row.ci_low)],[100*(row.ci_high-row.effect)]],
                    fmt="o",color="#b45720" if row.p_raw<.05 else "#353535",capsize=3)
    ax.axvline(0,c=".6",ls=":",lw=1)
    ax.set(yticks=range(len(models)),yticklabels=models.dimension,xlabel="Change in risk frequency (percentage points per unit score)")
    ax.invert_yaxis()
    return fig


def render(tables,out):
    folder=out/"figures"
    folder.mkdir(exist_ok=True)
    # Earlier releases wrote the pooled sixth panel a second time on its own.
    (folder / "pooled_all_split.pdf").unlink(missing_ok=True)
    def save(figure,name):
        figure.savefig(folder/f"{name}.pdf",bbox_inches="tight",metadata={"CreationDate": None})
        plt.close(figure)
    save(overall(tables["overall"]),"outcome_effects_split")
    save(state_panels(tables["state_counts"]),"categories_all_split")
    estimates=tables["mode_contrasts"]
    modes=sorted(estimates["mode"].unique())
    save(totals_figure(estimates,modes,["All"]),"all14_failure_case_totals")
    save(frequency(tables["task_risk_frequency"]),"risk_frequency_all_split_split_only")
    save(changes_figure(estimates,modes,by_category=True),"all14_case_incidence")
    names=tables["issue_inventory"].set_index("label").issue_type.to_dict()
    save(issue_panels(tables["issue_cells"], tables["outside_focal_modes"], names), "all14_codex_category_counts")
    for table,filename in [("issue_difference", "all14_codex_difference"),
                           ("role_cells", "all14_role_codex_category_counts"),
                           ("role_difference", "all14_role_codex_difference")]:
        ordering = tables["role_cells"] if table.startswith("role") else tables["issue_cells"]
        save(heatmaps(tables[table],ordering,"role" if table.startswith("role") else "label",names,
                      table.endswith("difference")),filename)
    save(feature_plot(tables["feature_models"]),"task_risk_frequency_all_split")
