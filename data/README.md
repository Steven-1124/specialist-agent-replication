# Data for the 30 tasks

The [task results](task_results.csv) give each task’s completion rate and mean number of flagged modes under each programmer condition, plus the fraction of its five groups that entered a risk state. The [original task requests](../experiment/tasks.csv) describe the software the teams were asked to build. Use `task_id` to follow a task across the files.

## Data used in the paper

| File | Observation and measure |
| --- | --- |
| [Task names](task_names.csv) | Display name and identifier for each of the 30 tasks. |
| [Task features](task_features.csv) | One scored task–feature pair (210), with its recorded rationale. |
| [Paired feature ratings](paired_feature_ratings.csv) | Two archived AI ratings for each of three features on each task (90 pairs). Add the two raw 0–2 ratings and divide by four to obtain the analyzed 0–1 score. |
| [Feature scoring rules](feature_rubric.csv) | Definitions, rating scales, anchors, and analyst-provenance notes for the seven features. |
| [Case states by task and group](published/cases.csv) | Outcome–process state and classification for each of 150 paired cases, one per task and group. |
| [Run-level inputs](executions.csv) | One row per run (900), identified by `group`, `task_id`, `arm`, and `run_number`. The completion indicator and each of the 14 failure-mode indicators record the [majority evaluator judgment](../votes/README.md). Source fields preserve generation and evaluator provenance. |
| [Checked issue associations](issue_associations.csv) | One checked association between an issue and a trace entry or event (6,131). The assigned primary role combines the CPO’s product and manual-writing stages under one role. |
| [Issue types](issue_types.csv) | The 58 retained types, including zero-count types, with their assigned MAST mode and focal role. |
| [Mode 2.6 trace review](mode_2_6_trace_recheck.csv) | One binary trace-support decision per run, used for the comparison with evaluator judgments in Section 5.1. |
| [Archived result tables](published/) | Independent targets for the [analysis script](../analysis/README.md), checked after the results have been computed. |
| [Source manifest](archive_sources.json) | Source filenames and SHA-256 hashes for the inputs and reference tables. |

## Inspecting an issue

Use the four run identifiers in an association row to open the [trace reader](../traces/README.md). The `trace_sequences` field identifies the relevant turns for 5,103 associations. The other 1,028 identify the run and issue type. When the evaluator nominated an issue, the archived pass, proposed mode, and index locate the nomination. The [issue-type registry](issue_types.csv) gives the mode assigned after review. Object and event numbers identify shared objects or events within each run. A blank event number means the archive did not record a distinct event.

Appendix A and the [coding rules](../coding/issue_rules.csv) give the criteria for retaining a checked association. One reviewed entry can contain several distinct behaviors. The archived decisions yield 3,180 generalist and 2,951 specialist associations. Independent review can use the linked traces, cited passages, and coding rules.
