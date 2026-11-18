# Archived evaluator judgments

The [task results](../data/task_results.csv) summarize completion and failure modes for each of the 30 tasks. The [underlying pass records](evaluator_passes.csv) give one completion decision and 14 failure-mode indicators for each evaluator pass. A value of one means the evaluator judged the task complete or flagged that mode, respectively.

To follow a task from its votes to the paper’s measures, select its `task_id` and group the rows by `group`, `arm` and `run_number`. Each run has three evaluator passes. A majority of two or three determines the run-level value of each indicator. Across the three runs within a group, task and programmer condition, the mean completion indicator gives the completion rate. The sum of the 14 mode frequencies gives the mean number of distinct flagged modes per run.

The archive contains 2,700 pass records for 900 runs. Each row carries hashes identifying its source completion and failure-mode responses. The [execution inventory](../data/executions.csv) preserves run-level provenance. Run `python3 verify.py` from the repository root to check the recorded decisions. The [analysis guide](../analysis/README.md) gives the estimators and paper outputs.
