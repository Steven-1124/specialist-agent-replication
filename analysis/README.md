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

## Estimation equations

### From evaluator votes to case outcomes

A case $i=(g,t)$ is one of five model groups $g$ crossed with one of 30 tasks $t$. Programmer assignment is $s=0$ for generalist and $s=1$ for specialist. Each arm has three executions $r$. For execution $isr$, write $c_{isrj}$ for evaluator pass $j$'s binary completion vote and $v_{isrjm}$ for its binary vote on MAST mode $m$. The root verification command checks each archived majority decision against the three pass records. Completion and each of the 14 distinct modes, including all five Category 1 modes, have separate majorities.

$$
C_{isr}=\mathbf{1}\!\left\{\sum_{j=1}^3 c_{isrj}\ge2\right\},
\qquad M_{isrm}=\mathbf{1}\!\left\{\sum_{j=1}^3 v_{isrjm}\ge2\right\}.
$$

$$
C_{is}=\frac{1}{3}\sum_{r=1}^3 C_{isr},\qquad
f_{ism}=\frac{1}{3}\sum_{r=1}^3 M_{isrm},\qquad
F_{is}=\sum_{m=1}^{14} f_{ism}.
$$

$C_{is}$ is the completion rate, $f_{ism}$ the proportion of executions flagged for mode $m$, and $F_{is}$ the mean number of distinct flagged modes per execution (range 0–14). An execution may contribute to several modes.

### Paired estimates and intervals

For $Y=C$ or $F$, define $\Delta_iY=Y_{i1}-Y_{i0}$. The implementation directly averages these changes. Equivalently, in the balanced case-arm regression below, $\alpha_i$ absorbs the group–task case level and $\beta_Y$ is the specialist-minus-generalist coefficient.

$$
Y_{is}=\alpha_i+\beta_Y s+\varepsilon_{is},\qquad
\widehat\beta_Y=\frac{1}{n}\sum_i \Delta_iY,\qquad
\widehat{\mathrm{SE}}(\widehat\beta_Y)=
\sqrt{\frac{\sum_i(\Delta_iY-\widehat\beta_Y)^2}{n(n-1)}}.
$$

The 95% interval is $\widehat\beta_Y\pm t_{.975,n-1}\widehat{\mathrm{SE}}$. The two-sided test uses the same $n-1$ degrees of freedom. Each case has equal weight, with $n=150$ pooled or $n=30$ within any group. Pooled completion rises from 32.44% to 48.89%, a change of 16.44 percentage points (95% CI 10.99 to 21.90). Mean burden changes by −0.242 modes per execution (95% CI −0.422 to −0.063).

### State assignment, task frequency, and flag composition

The signs of $(\Delta_i C,\Delta_i F)$ locate each case in a nine-cell table. The code compares integer three-run totals to preserve exact ties. Define $H_i=R$ if $\Delta_i C\ge0$ and $\Delta_i F>0$, and $H_i=B$ if $\Delta_i C\ge0$ and $\Delta_i F<0$. The two risk cells contain 35 cases, the two benefit cells 74, and the remaining five cells 41. A group-specific cell share divides its case count by 30 and the pooled share divides by 150. The task frequency counts risk cases among five groups, while the arm-by-mode total counts flagged executions across all cases.

$$
q_t=\frac{1}{5}\sum_{g=1}^5\mathbf{1}\{H_{(g,t)}=R\},\qquad
N_{sm}=\sum_{i=1}^{150}\sum_{r=1}^{3} M_{isrm}.
$$

For example, Gold Miner has $q_t=5/5=1$. The composition figures rank modes by $N_{0m}+N_{1m}$. A pooled mode's share divides that sum by 2,405 flags (1,257 generalist plus 1,148 specialist). Category totals sum over member modes.

### Within-state changes in mode occurrence

For mode $m$, $d_{im}=3(f_{i1m}-f_{i0m})$ counts the change in flagged executions per case. Each arm can contribute zero through three. For $H=R,B$, let $n_H$ be its case count. The same mean and paired interval also apply across all 150 cases.

$$
\bar d_{Hm}=\frac{1}{n_H}\sum_{i:H_i=H} d_{im},\qquad
\widehat\gamma_m=\bar d_{Rm}-\bar d_{Bm}.
$$

Within-state intervals use the paired-case formula above. For $\widehat\gamma_m$, let $a_H=s^2_{Hm}/n_H$, where $s^2_{Hm}$ is the sample variance of $d_{im}$ within state $H$. The two-sided Welch test and 95% interval use the following standard error and degrees of freedom.

$$
\mathrm{SE}(\widehat\gamma_m)=\sqrt{a_R+a_B},\qquad
\nu=\frac{(a_R+a_B)^2}{a_R^2/(n_R-1)+a_B^2/(n_B-1)},\qquad
\widehat\gamma_m\pm t_{.975,\nu}\mathrm{SE}(\widehat\gamma_m).
$$

For mode 2.6, $\bar d_{Rm}=1.029$, $\bar d_{Bm}=-0.676$ and $\widehat\gamma_m=1.704$ flagged executions per three. The sign of total flag change helps define the states, so these contrasts describe how the modes contribute to the observed divergence.

### Checked issues, roles, and trace recheck

Every checked adverse association $a$ belongs to an execution, issue type, primary mode, and primary role. For issue type $\ell$ and mode $m$, the cell count, within-state change, and risk-minus-benefit difference are

$$
A_{Hsm\ell}=\sum_a \mathbf{1}\{H_{i(a)}=H,\ s(a)=s,\ \ell(a)=\ell,\ m(a)=m\},\qquad
\Delta A_{Hm\ell}=A_{H1m\ell}-A_{H0m\ell},\qquad
D_{m\ell}^{A}=\Delta A_{Rm\ell}-\Delta A_{Bm\ell}.
$$

The role panels replace $\ell(a)$ with the assigned primary role. These are raw counts over 35 risk or 74 benefit cases. A single execution can have several checked associations. The inventory uses all states and both arms. An issue type's share equals its total count divided by all 6,131 associations. Issue types above 0.1% get separate rows within the seven focal modes selected by the unadjusted Welch $p<.05$ criterion. The main issue table also retains an aggregate outside the focal modes.

For the mode-2.6 trace recheck, $z_{isr}$ is either the standard binary mode vote or the final binary trace-supported indicator (`rechecked_support`). Set $d_i^z=\frac{100}{3}\sum_r(z_{i1r}-z_{i0r})$. The reported contrast is $\bar d_R^z-\bar d_B^z$ with the same Welch inference and the original state assignments. It is 56.81 percentage points for the standard vote and 14.03 for final trace support (95% CI −3.13 to 31.19). The preliminary residual-screen field records an earlier review decision.

### Task-feature estimates and held-out prediction

For each feature $k$, fit a separate ordinary least squares model to the 30 equally weighted tasks. For the three features with two archived ratings on a 0–2 scale, $x_{tk}=(r_{1tk}+r_{2tk})/4$. Each of the other four scores is a single analyst judgment on the 0–1 scale. Prior outcome exposure and the absence of independent human validation constrain interpretation. The model and slope are

$$
q_t=\alpha_k+\theta_k x_{tk}+u_{tk},\qquad
\widehat\theta_k=\frac{\sum_t(x_{tk}-\bar x_k)(q_t-\bar q)}
{\sum_t(x_{tk}-\bar x_k)^2}.
$$

$100\widehat\theta_k$ is the fitted difference in risk frequency, in percentage points, between scores of zero and one. With design matrix $X=[\mathbf{1},x_k]$, residuals $e_t$ and hat diagonals $h_t$, the HC3 covariance is $(X^\top X)^{-1} X^\top \operatorname{diag}[e_t^2/(1-h_t)^2] X (X^\top X)^{-1}$. The slope interval and two-sided test use 28 degrees of freedom. The feature-correlation table reports Pearson correlations between score columns across the same 30 tasks.

For each held-out task $t$, refit the model on the other 29 and form its prediction $\widehat q_{tk}^{(-t)}$. The comparison prediction $\bar q^{(-t)}$ is the training-fold mean. A negative value indicates larger squared prediction error than that mean across folds.

$$
R^2_{\mathrm{LOTO},k}=1-
\frac{\sum_t (q_t-\widehat q_{tk}^{(-t)})^2}
{\sum_t (q_t-\bar q^{(-t)})^2}.
$$

## Numerical checks

`reproduce.py` checks individual case identities and states, task frequencies for all 30 tasks, each arm's completion and flag means and paired contrasts, all 56 mode-contrast rows, the inventory of 58 issue types, the published focal-issue table, every defined issue- and role-plot cell, seven task-feature regressions, and both mode-2.6 recheck estimates. Count and classification results must agree exactly. Numeric fits must agree within the script's floating-point tolerances.

The input data dictionary and archive provenance are in [data/README.md](../data/README.md). Archival t interval, Welch and HC3/held-out-task implementations are retained in `inference.py`. The original paired-mode and failure-total plotting routines are retained in `mode_plots.py`.
