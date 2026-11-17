# Using a Specialist Agent in a Team of Agents

Replication materials for *Using a Specialist Agent in a Team of Agents: Outcome and Process in Software Development*, by Yanxuan (Steve) Wu and Tingting Nian.

What changes when a software-development team keeps its workflow, task and surrounding agents fixed, but replaces its generalist programmer with a coding-specialized model? The experiment crosses 30 ProgramDev tasks with five groups. Each group runs each task three times in each of the two programmer conditions, for 900 executions and 150 paired group–task cases. All analyses use the full 14-mode MAST taxonomy, including Category 1.

## Verify the main result

Clone the repository and run the following with Python 3.10 or later. This check uses the Python standard library and the saved evaluator decisions. It runs offline and requires no API key or model weights.

```sh
git clone https://github.com/Steven-1124/specialist-agent-replication.git
cd specialist-agent-replication
python3 verify.py
```

| Standard-split measure | Generalist programmer | Specialist programmer |
| --- | ---: | ---: |
| Executions judged complete | 146 / 450 (32.44%) | 220 / 450 (48.89%) |
| Mean number of distinct failure modes flagged per execution | 2.79 | 2.55 |

The verifier reconstructs every execution's majority decision from its three evaluator passes. In 35 of 150 paired cases (23.3%), completion held steady or improved while the number of flagged modes increased. These cases span 21 of the 30 tasks. A task's risk frequency is the fraction of its five groups that show this pattern.

## Reproduce the tables and figures

With CPython 3.10 installed, run the following from the repository root.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
```

The command reconstructs the numerical tables and 11 empirical figures in `build/analysis/` and checks calculated values against archived results. It covers case classifications, task risk frequency, failure mode incidence, checked issues and their associated roles, and task-feature regressions. The [analysis guide](analysis/README.md) maps the outputs to the paper and explains the estimators.

## Inspect the evidence or run a new experiment

| Material | What it contains |
| --- | --- |
| [Study design](design/) | The three schematic exhibits from the paper, showing the design, agent workflow and nine possible case states. |
| [Evaluator passes](votes/README.md) | 2,700 paired completion and failure judgments, including the original and recovery evaluation settings. |
| [Analysis inputs](data/) | 900 majority-voted executions, 6,131 retained issue associations, task ratings and archived result tables. |
| [Agent traces](traces/README.md) | A searchable archive of the 900 executions and the available evidence behind individual issue assignments. The guide documents the release redactions. |
| [Coding rules](coding/issue_rules.csv) | Definitions, inclusion criteria, exclusion criteria and the primary MAST mode for each of the 58 retained issue types. Five entries summarize the later trace review. |
| [Prompts](prompts/README.md) | The six agent-role templates and the three evaluator templates, with their SHA-256 checksums. |
| [Experimental workflow](experiment/README.md) | The 30 task requests, five model configurations, run schedule, model-serving code and commands for replay, new generation and evaluation. |

The original case assignments and anchored passages can be checked with `python3 traces/verify.py`. In 243 runs the public trace masks at least one string, such as a contact address or credential-like literal. The run-level numerical data cover all 900 executions.

The archived analysis treats the study's trace reviews and task-feature ratings as recorded judgments. Readers can check them against the traces and coding rules or conduct an independent review. Fresh model calls require model access and compute and may incur provider charges. Sampling and historical evaluator backend changes can produce different answers on rerun.

## Sources

The workflow adapts [ChatDev](https://github.com/OpenBMB/ChatDev). The task design and failure taxonomy build on Cemri and colleagues' [MAST study](https://arxiv.org/abs/2503.13657) and [released materials](https://github.com/multi-agent-systems-failure-taxonomy/MAST/tree/a70542e541b2104ef8fcd785778179e173fb8d70). The included [ChatDev license](prompts/ChatDev-LICENSE) and the attributions in the component guides accompany the corresponding third-party materials.
