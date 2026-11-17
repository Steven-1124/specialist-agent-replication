# The original standard-split judgments

`standard_split_passes.csv` contains 2,700 rows. Each row pairs the two independent judgments used for one evaluator pass over one saved execution. A row has one completion flag and a separate binary flag for each of the 14 MAST modes, including all five Category 1 modes. A value of 1 means the evaluator flagged the mode or judged the task complete.

The key is `group, task_id, arm, run_number, evaluator_pass`. Each of the 900 executions has exactly three passes. Take the majority independently for each of the 15 columns. Sum the resulting 14 mode flags to calculate the number of distinct flagged modes per execution. Then average these run-level quantities across the three runs for each group, task and arm. The paper's risk classification compares the two arms within a group and task using all 14 modes.

Two digest columns identify the separate archived completion and failure response records. At release preparation, both response records for each row were hash-checked against the study registry and the saved trace. The flags were parsed from the original returned text and compared with the stored parsed judgments. The public CSV contains the binary results of these checks.

The same DeepSeek Flash response alias appears in the archive under three documented serving regimes. The original 8,192-token setting covers 785 executions. Of the 115 executions with recovered judgments using a 262,144-token cap, 108 used the Flash-1017 regime and seven have an unpinned historical backend. Each execution has three passes under its recorded regime. The P2 issue-identification evaluator furnished additional evidence for the mechanism analysis. Its nominations are a distinct measurement from the standard-split votes in this file.

From the repository root, run `python3 verify.py` to reconstruct and check the majority decisions and the paper's headline statistics from these votes. See the [analysis guide](../analysis/README.md) for the full set of estimates.
