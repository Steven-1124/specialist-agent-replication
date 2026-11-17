"""Recompute the all-14-mode paper from archived run labels and issue decisions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inference import interval, welch, slope

ROOT = Path(__file__).resolve().parents[1]
CASE = ["group", "task_id"]
RUN = CASE + ["arm", "run_number"]
MODES = [f"{c}.{i}" for c, n in ((1, 5), (2, 6), (3, 3)) for i in range(1, n+1)]
FLAGS = ["mode_" + m for m in MODES]
ROLES = ["CEO", "CPO", "CTO", "Coder", "Reviewer"]
STATES = [
    ["Clean win", "Completion gain", "Mirage success"],
    ["Process cleanup", "Status quo", "Hidden deterioration"],
    ["Compensating decline", "Completion loss", "Joint deterioration"],
]
DIFFERENCE = "Risk minus Benefit"


def case_results(runs):
    assert len(runs) == 900 and not runs.duplicated(RUN).any()
    assert runs.groupby(CASE + ["arm"]).size().eq(3).all()
    assert set(runs.run_number) == {1, 2, 3}
    assert runs[FLAGS + ["completion"]].isin([0, 1]).all().all()
    runs = runs.assign(flags=runs[FLAGS].sum(axis=1))
    totals = runs.groupby(CASE + ["arm"])[["completion", "flags"]].sum().unstack()
    changes = totals.xs("specialist", axis=1, level=1) - totals.xs("generalist", axis=1, level=1)
    states = np.asarray(STATES)[1 - np.sign(changes.completion), 1 + np.sign(changes["flags"])]
    cases = changes.rename(columns={"completion": "completion_change_3_runs", "flags": "flag_change_3_runs"}).assign(state=states)
    cases["category"] = np.select(
        [cases.state.isin(["Mirage success", "Hidden deterioration"]), cases.state.isin(["Clean win", "Process cleanup"])],
        ["Risk", "Benefit"], default="Other"
    )
    cases = cases.reset_index()
    assert len(cases) == 150 and cases.groupby("task_id").size().eq(5).all()
    overall = []
    for group in ["Pooled", *sorted(runs.group.unique())]:
        part = totals if group == "Pooled" else totals.xs(group)
        for metric in ("completion", "flags"):
            levels = part[metric] / 3
            overall.append(dict(group=group, metric=metric, generalist=levels.generalist.mean(),
                                specialist=levels.specialist.mean(), **interval(levels.specialist - levels.generalist)))
    states_by_group = pd.crosstab(cases.group, cases.state).reindex(columns=sum(STATES, []), fill_value=0)
    states_by_group.loc["Pooled"] = states_by_group.sum()
    frequency = cases.pivot(index="task_id", columns="group", values="state")
    frequency["risk_groups"] = cases.category.eq("Risk").groupby(cases.task_id).sum()
    frequency["risk_frequency"] = frequency.risk_groups / 5
    return cases, pd.DataFrame(overall), states_by_group.reset_index(), frequency.reset_index()


def mode_results(runs, cases):
    labels = runs.melt(id_vars=RUN, value_vars=FLAGS, var_name="mode", value_name="flag")
    labels["mode"] = labels["mode"].str[5:]
    counts = labels.groupby(CASE + ["mode", "arm"]).flag.sum().unstack().reset_index()
    counts = counts.merge(cases[CASE + ["category"]], on=CASE, validate="many_to_one")
    counts["change"] = counts.specialist - counts.generalist
    rows = []
    for mode, frame in counts.groupby("mode"):
        risk = frame[frame.category.eq("Risk")]
        benefit = frame[frame.category.eq("Benefit")]
        for contrast, group in [("All", frame), ("Risk", risk), ("Benefit", benefit)]:
            rows.append(dict(mode=mode, contrast=contrast, generalist=group.generalist.sum(),
                             specialist=group.specialist.sum(), **interval(group.change)))
        rows.append(dict(mode=mode, contrast=DIFFERENCE, **welch(risk.change, benefit.change)))
    return pd.DataFrame(rows)


def issue_results(issues, definitions, cases, modes):
    merged = issues.merge(definitions, on="label", validate="many_to_one")
    assert len(merged) == len(issues) == 6131
    # A single involved role controls. Ambiguous or missing roles inherit the issue type's actor.
    involved = merged.involved_roles.fillna("")
    resolved = involved.where(involved.isin(ROLES), merged.type_subject_role)
    assert resolved.eq(merged.role).all()
    merged["role"] = resolved
    merged = merged.merge(cases[CASE + ["category"]], on=CASE, validate="many_to_one")
    inventory = merged.groupby(["label", "arm"]).size().unstack(fill_value=0).reindex(definitions.label, fill_value=0)
    inventory["change"] = inventory.specialist - inventory.generalist
    inventory["occurrence_share"] = (inventory.generalist + inventory.specialist) / len(merged)
    inventory = definitions.merge(inventory.reset_index(), on="label", validate="one_to_one")
    diff = modes[(modes.contrast == DIFFERENCE) & (modes.p_raw < .05)]
    focal = diff.sort_values("estimate", ascending=False)["mode"].tolist()
    named = inventory[(inventory.occurrence_share > .001) & inventory["mode"].isin(focal)].label
    merged["display_label"] = merged.label.where(merged.label.isin(named), "Other issues")
    summary = merged.copy()
    summary.loc[~summary["mode"].isin(focal), ["mode", "display_label"]] = ["Outside focal modes", "Other issues"]
    display = summary.groupby(["mode", "display_label", "arm"]).size().unstack(fill_value=0)
    display["change"] = display.specialist - display.generalist
    display["occurrence_share"] = (display.generalist + display.specialist) / len(merged)
    selected = merged[merged["mode"].isin(focal) & merged.category.isin(["Risk", "Benefit"])]
    rows_issue, rows_role = [], []
    for mode in focal:
        component = selected[selected["mode"] == mode]
        local_names = list(inventory[inventory.label.isin(named) & inventory["mode"].eq(mode)].label)
        for category in ("Risk", "Benefit"):
            sample = component[component.category == category]
            for role in ROLES:
                arms = sample.loc[sample.role.eq(role), "arm"].value_counts()
                rows_role.append(dict(category=category, role=role, mode=mode,
                                      raw_change=arms.get("specialist", 0)-arms.get("generalist", 0)))
            for label in local_names + ["Other issues", "Focal mode total"]:
                subset = sample if label == "Focal mode total" else sample[sample.display_label == label]
                arms = subset.arm.value_counts()
                rows_issue.append(dict(category=category, label=label, mode=mode,
                                       raw_change=arms.get("specialist",0)-arms.get("generalist",0)))
    def contrasts(rows, key):
        cells = pd.DataFrame(rows)
        arms = cells.pivot(index=[key,"mode"], columns="category", values="raw_change")
        between = (arms.Risk - arms.Benefit).rename("raw_change").reset_index().assign(category=DIFFERENCE)
        return cells, between
    return inventory, display.reset_index(), *contrasts(rows_issue,"label"), *contrasts(rows_role,"role")


def feature_results(scores, ratings, frequency):
    scores = scores.merge(ratings, how="left", on=["task_id","dimension"], validate="one_to_one")
    paired = scores.rater_1.notna()
    assert paired.sum() == 90 and scores.rater_1.isna().equals(scores.rater_2.isna())
    rescaled_pair = (scores.loc[paired, "rater_1"] + scores.loc[paired, "rater_2"]) / 4
    np.testing.assert_allclose(rescaled_pair, scores.loc[paired, "score"])
    scores.loc[paired, "score"] = rescaled_pair
    models = []
    for dimension, group in scores.groupby("dimension", sort=False):
        frame = group.merge(frequency[["task_id", "risk_frequency"]], on="task_id", validate="one_to_one")
        assert len(frame) == 30 and frame.score.between(0,1).all()
        models.append(dict(dimension=dimension, **slope(frame.score, frame.risk_frequency)))
    correlation = scores.pivot(index="task_id", columns="dimension", values="score").corr(method="pearson")
    return pd.DataFrame(models), correlation.rename_axis("dimension").reset_index()


def trace_recheck(flags, cases, runs):
    merged = flags.merge(runs[RUN + ["mode_2.6"]], on=RUN, validate="one_to_one")
    assert len(merged) == len(runs)
    rows = []
    for metric in ["standard_vote", "rechecked_support"]:
        source = "mode_2.6" if metric == "standard_vote" else metric
        levels = merged.groupby(CASE + ["arm"])[source].mean().unstack()
        changes = (100*(levels.specialist - levels.generalist)).rename("delta").reset_index().merge(cases, on=CASE)
        risk = changes.loc[changes.category == "Risk", "delta"]
        benefit = changes.loc[changes.category == "Benefit", "delta"]
        rows.append(dict(source=metric, risk_change=risk.mean(), benefit_change=benefit.mean(), **welch(risk, benefit)))
    return pd.DataFrame(rows)


def read(data, name):
    return pd.read_csv(data / f"{name}.csv", dtype={"task_id":str, "mode":str})


def verify(result, data):
    keys = {
        "cases":CASE, "overall":["group","metric"], "mode_contrasts":["mode","contrast"],
        "task_risk_frequency":["task_id"], "focal_issue_table":["mode","display_label"],
        "issue_inventory":["label"], "issue_cells":["category","label","mode"],
        "issue_difference":["category","label","mode"], "role_cells":["category","role","mode"],
        "role_difference":["category","role","mode"], "recheck":["source"], "feature_models":["dimension"],
    }
    for name, id_cols in keys.items():
        expected = read(data / "published", name).set_index(id_cols).sort_index()
        actual = result[name].set_index(id_cols).sort_index()[expected.columns]
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_names=False,
                                      rtol=1e-9, atol=1e-11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--out", type=Path, default=ROOT/"build"/"analysis")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs = read(args.data, "executions")
    r = {}
    r["cases"],r["overall"],r["state_counts"],r["task_risk_frequency"] = case_results(runs)
    r["task_risk_frequency"] = r["task_risk_frequency"].merge(read(args.data,"task_names"), on="task_id", validate="one_to_one")
    r["mode_contrasts"] = mode_results(runs,r["cases"])
    all_modes = r["mode_contrasts"].query("contrast == 'All'").copy()
    all_modes["combined_flags"] = all_modes.generalist + all_modes.specialist
    r["mode_ranking"] = all_modes.sort_values("combined_flags",ascending=False)
    r["flags_by_category"] = all_modes.groupby(all_modes["mode"].str[0])[["generalist","specialist","combined_flags"]].sum().rename_axis("mast_category").reset_index()
    (r["issue_inventory"],r["focal_issue_table"],r["issue_cells"],r["issue_difference"],
     r["role_cells"],r["role_difference"]) = issue_results(
        read(args.data,"issue_associations"),read(args.data,"issue_types"),r["cases"],r["mode_contrasts"])
    r["feature_models"],r["feature_correlations"] = feature_results(read(args.data,"task_features"),read(args.data,"paired_feature_ratings"),r["task_risk_frequency"])
    r["recheck"] = trace_recheck(read(args.data,"mode_2_6_trace_recheck"),r["cases"],runs)
    verify(r,args.data)
    args.out.mkdir(parents=True,exist_ok=True)
    for name,frame in r.items():
        frame.to_csv(args.out / f"{name}.csv",index=False)
    if not args.no_plots:
        from plots import render
        render(r,args.out)
    print(f"Verified {len(runs)} executions, {len(r['cases'])} cases and {len(r['issue_inventory'])} issue types against the published analysis.")
    print(r["cases"].category.value_counts().to_string())
    print(f"Wrote numerical results to {args.out}")


if __name__ == "__main__":
    main()
