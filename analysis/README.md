# Reproduce the paper's quantitative results

From the repository root, use CPython 3.10 and run the following commands. Subsequent analysis runs work offline. The analysis uses the temperature-1 standard split evaluator and all 14 MAST modes, including Category 1.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
```

Outputs go to `build/analysis/`. Use `--out PATH` for another directory or `--no-plots` for the numerical tables alone. The script first calculates the results from the individual runs, checked issue associations, and task ratings. It then compares them with 12 independently exported tables in [the published-result archive](../data/published/). The same run writes the ten empirical figures as PDFs. The numerical tables preserve full precision. The regenerated plots use the paper's measures and data, with some differences in presentation.

| Working-paper exhibit | Output in `build/analysis/` |
| --- | --- |
| Figures 1–3 and Tables 1–2 | See the [three design exhibits](../design/), the [six agent contracts](../prompts/mas/), and the [model assignments](../experiment/models.json). |
| Table 3 | In `data/executions.csv`, count rows by group and arm. Each group has 90 executions per arm and 30 distinct tasks. |
| Figure 4, overall changes | `overall.csv` and `figures/outcome_effects_split.pdf` |
| Figure 5, outcome and process combinations | `cases.csv`, `state_counts.csv`, and `figures/categories_all_split.pdf` (five groups and the pooled sample) |
| Figure 6 and Table 4, failure-mode composition | `mode_ranking.csv`, `flags_by_category.csv`, and `figures/all14_failure_case_totals.pdf` |
| Figure 7, task heterogeneity | `task_risk_frequency.csv` and `figures/risk_frequency_all_split_split_only.pdf` |
| Figure 8 and Table 5, within-state changes and their contrast | `mode_contrasts.csv` and `figures/all14_case_incidence.pdf` |
| Figure 9 and Table 6, issue-type composition within each state | `focal_issue_table.csv`, `issue_cells.csv`, `outside_focal_modes.csv`, and `figures/all14_codex_category_counts.pdf` (risk and benefit panels side by side) |
| Figure 10, risk-minus-benefit difference in issue counts | `issue_difference.csv` and `figures/all14_codex_difference.pdf` |
| Figures 11–12, role composition | `role_cells.csv`, `role_difference.csv`, and both corresponding role figures |
| Section 5.1, evaluator and issue-supported mode 2.6 | `recheck.csv` |
| Figure 13 and Table 7, task features | `feature_models.csv`, `feature_correlations.csv`, `figures/task_risk_frequency_all_split.pdf`, and the scoring instructions in `data/feature_rubric.csv` |
| Appendix Tables 8–9 | `issue_inventory.csv`, including observed counts, zero-count types, and primary-mode assignments |

## Analysis units

- Each of 150 cases is a group crossed with a ProgramDev task. Within each case the generalist and specialist programmers each generated three executions. Completion and 14 binary flags are already majority decisions over three evaluator passes per execution. The root verification command checks the run-level votes against the archived passes. Equal-case comparisons give each of the 150 cases one observation.
- For Figure 4, calculate the mean completion or distinct-mode count in each arm of each case. Subtract the generalist mean from the specialist mean. Calculate a paired-case t interval and two-sided t test. Completion values in the CSV are proportions. Multiply by 100 to obtain percentage-point differences.
- For cell assignment, compare the integer completion totals and total flagged modes over the three runs in each arm. A risk case has more flags with flat or higher completion. A benefit case has fewer flags with flat or higher completion. The other five combinations have their own state names. Each task's risk frequency divides its count of risk cases by five. All five groups have data on every task.
- The mechanism comparison uses per-case counts across three runs, giving a zero-to-three range for each individual mode and arm. Within-state intervals use a paired-case t calculation. The risk-minus-benefit contrast uses Welch's independent-sample t procedure. The states depend partly on the counts under study. The contrast describes where the divergence occurs.
- Each checked adverse association contributes once to an issue type, primary mode, and primary role. Raw issue and role panels calculate changes within each state, then differences between those changes. The 35 risk cases and 74 benefit cases have unequal sample sizes. A focal mode has an unadjusted contrast p-value below .05. Within one of these modes, issue types that exceed 0.1% of all 6,131 associations receive separate rows. The main issue table retains an aggregate outside the focal modes. Each plotted row total can be regenerated directly from the association ledger.
- The trace recheck uses a binary indicator per execution, computed separately for a standard vote and trace-supported mode 2.6. Within an arm, average the three indicators, form the paired change, then compare risk and benefit cases in percentage points. The published recheck corresponds to `rechecked_support` in the final disagreement review; the preliminary residual-screen measure is a different indicator.
- Seven separate OLS regressions predict the 30 task-level risk frequencies from scores in [0, 1]. Uncertainty uses HC3 standard errors with t critical values. Each held-out task's error is compared with that of the training-set mean for that fold. Three of the seven feature scores are reconstructed from paired archived ratings on the original 0–2 scale. Each of the four other scores is a single analyst judgment. Prior outcome exposure and the absence of independent human validation constrain interpretation.

## Numerical checks

`reproduce.py` checks individual case identities and states, task frequencies for all 30 tasks, each arm's completion and flag means and paired contrasts, all 56 mode-contrast rows, the inventory of 58 issue types, the published focal-issue table, every defined issue- and role-plot cell, seven task-feature regressions, and both mode-2.6 recheck estimates. Count and classification results must agree exactly. Numeric fits must agree within the script's floating-point tolerances.

The input data dictionary and archive provenance are in [data/README.md](../data/README.md). Archival t interval, Welch and HC3/held-out-task implementations are retained in `inference.py`. The original paired-mode and failure-total plotting routines are retained in `mode_plots.py`.
