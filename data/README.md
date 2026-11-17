# Archived analysis inputs

The common execution key is `group, task_id, arm, run_number`. Groups run from G1 through G5. The arm is `generalist` or `specialist` and run number ranges from 1 to 3. The task ID is the archived ProgramDev identifier. These fields link the votes, checked issue associations, mode recheck, and execution records. A group–task case contains six executions.

| File | Unit and interpretation |
| --- | --- |
| `executions.csv` | One row per execution (900). Completion and the 14 `mode_*` fields are majority votes, each 0 or 1. Generation source and judge regime preserve the original provenance. There are 785 original runs and 115 recovery runs. The original regime covers 785 runs, the dated Flash-1017 regime covers 108, and seven use the native-context regime with an unpinned dated backend. |
| `issue_associations.csv` | One retained adverse entry–type association or trace-rule occurrence per row (6,131). It carries its issue code, the set of recognized involved roles after CPO manual is folded into CPO, and the archived primary role. A single involved role controls. If roles are absent or multiple, the issue type supplies its focal actor. See Appendix A for counting rules. |
| `issue_types.csv` | Complete registry of the 58 retained adverse issue types, including types observed zero times. The primary mode is the final evidence-qualified classification. The type_subject_role column gives the fallback actor for the issue type. |
| `mode_2_6_trace_recheck.csv` | One final issue-supported binary flag per execution after rechecking mode-2.6 disagreements (900). Its distinction from the initial standard vote is used in Section 5.1. |
| `task_features.csv` | One scored task–feature pair (210). Includes its recorded reason. Some features share a task-level rationale. |
| `paired_feature_ratings.csv` | Both archived AI ratings for each of three features on 30 tasks (90 pairs). Add the two raw 0–2 ratings and divide by four for the mean on the 0–1 scale used in the regressions. |
| `feature_rubric.csv` | Seven prompts, original rating scales, coding anchors, and analyst-provenance notes. |
| `task_names.csv` | The display name for each task identifier. |
| `published/` | Independent numerical targets exported directly from the archived full-mode tables. The replication reads them only after it has computed the results. |
| `archive_sources.json` | Original project-relative source names and SHA-256 hashes of the archive files used for the projections and verification tables. |

## Auditing issue decisions

The [trace reader](../traces/README.md) opens the archived outputs and evidence attached to an association. The [coding rules](../coding/issue_rules.csv) document the issue definitions. To inspect an association, select its execution by the four key fields, then examine the recorded trace steps in `trace_sequences` (semicolon-separated original sequence numbers). These are populated on 5,103 rows. The remaining 1,028 rows retain their execution and issue type. Where the source recorded a P2 nomination, the pass, proposed failure mode and nomination index are retained. The P2-proposed mode and final checked primary mode can differ. Look up the latter using `issue_types.csv`. Blank nomination fields identify associations without that locator. `object_number` and `event_number` are sequential identifiers within an execution, preserving shared source objects and events without publishing machine-specific filenames. A blank event number means no distinct event ID was recorded.

The association ledger contains the final accepted adverse decisions used in the analysis. Each analyzed type has one primary MAST destination, although a single reviewed object can supply several distinct behaviors. The published totals are 3,180 generalist and 2,951 specialist associations. Automatic regeneration of these semantic judgments from scratch would require a new review of the traces. Holding the archived decisions fixed reproduces all reported counts. A new review can use the trace locators, prompts, counting rules and issue definitions to evaluate agreement with those decisions.
