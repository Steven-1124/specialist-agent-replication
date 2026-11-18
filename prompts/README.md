# Agent and evaluator prompts

Yanxuan (Steve) Wu and Tingting Nian

These files contain the six role templates used in the software-development workflow and the three evaluator templates used to study its outcomes and process. The programmer receives the same role template in the generalist and specialist conditions. The chief product officer appears twice in the workflow, once to specify requirements and once to prepare the manual.

| Role or evaluation | Template |
| --- | --- |
| Chief executive officer | [Performance criteria and constraints](mas/01_ceo.txt) |
| Chief product officer | [Product requirements](mas/02_cpo.txt) |
| Chief technology officer | [Implementation plan](mas/03_cto.txt) |
| Programmer | [Implementation and revision](mas/04_programmer.txt) |
| Code reviewer | [Code review](mas/05_reviewer.txt) |
| Chief product officer | [Final manual](mas/06_cpo_manual.txt) |
| Task completion evaluator | [Binary completion judgment](mast/01_task_completion.txt) |
| Failure-mode evaluator | [MAST failure modes](mast/02_failure_modes.txt) |
| Issue-identification evaluator | [Issue identification and trace evidence](mast/03_issue_identification.txt) |

The role templates include the shared company instruction. At execution, the workflow fills the task and handoff placeholders with the customer request and the relevant saved outputs. The evaluator files use `[EXECUTION-SPECIFIC MAS TRACE]` for the trace under review. The templates retain the taxonomy definitions and examples presented to the evaluator.

The role prompts adapt [ChatDev](https://github.com/OpenBMB/ChatDev). The included [Apache license](ChatDev-LICENSE) accompanies the ChatDev material. The definitions and historical examples in the two failure-mode evaluators come from [Cemri and colleagues’ MAST materials](https://github.com/multi-agent-systems-failure-taxonomy/MAST/tree/a70542e541b2104ef8fcd785778179e173fb8d70). [SHA256SUMS](SHA256SUMS) records the SHA-256 digest of each of the nine templates, including its original line breaks.

The associated [MAST-Data dataset](https://huggingface.co/datasets/mcemri/MAST-Data) is attributed to Cemri and colleagues under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The study combines the upstream material with its own judge instructions.
