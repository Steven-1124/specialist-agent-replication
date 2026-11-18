# Better Outcome, Worse Process? The Hidden Cost of Using a Specialist Agent in an Agentic Team

Replication materials for the working paper by Yanxuan (Steve) Wu and Tingting Nian.

## Results by task

Five teams each worked on the same 30 software requests with a generalist programmer and with a specialist programmer. The teams kept their other roles fixed. Each task has 15 runs per programmer condition, three from each team. Across tasks, judged completion rose from 32.44% to 48.89%, and the mean number of distinct failure modes flagged per run fell from 2.79 to 2.55.

**[Browse results for all 30 tasks](data/task_results.csv)** · [Read the original task requests](experiment/tasks.csv) · [See each task’s outcome–process state in each team](data/published/task_risk_frequency.csv)

| Task and requested software | Judged completion, generalist → specialist | Flagged modes per run, generalist → specialist | Teams in a risk state |
| --- | ---: | ---: | ---: |
| Gold Miner — time a moving claw to collect gold before the level ends | 13.33% → 26.67% | 2.13 → 3.20 | 5 of 5 |
| The Crossword — enter words by clue and direction and check intersecting letters | 6.67% → 60.00% | 2.87 → 2.47 | 3 of 5 |
| Sudoku — fill a 9×9 grid, identify mistakes, and check the completed puzzle | 40.00% → 60.00% | 2.73 → 1.87 | 0 of 5 |

A team enters a risk state on a task when switching to the specialist leaves completion unchanged or higher and increases the mean number of flagged modes. For The Crossword, the average flag count fell, yet three teams entered a risk state. We classify each of the 150 team–task pairs before counting how often risk occurs for each task. This pattern appears in 35 pairs (23.3%), spanning 21 tasks.

## Reproduce the results

To verify and print the 30 task summaries from the archived judgments, run the following with Python 3.10 or later. The check runs offline after cloning.

```sh
git clone https://github.com/Steven-1124/specialist-agent-replication.git
cd specialist-agent-replication
python3 verify.py
```

To rebuild the paper’s numerical tables and ten empirical figures, run the following from the repository root. The results are written to `build/analysis/` and checked against the archived tables. [Analysis guide and exhibit index](analysis/README.md)

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
```

The [experimental workflow](experiment/README.md) and [prompts](prompts/README.md) document how the teams produced the saved work. The workflow adapts [ChatDev](https://github.com/OpenBMB/ChatDev). The tasks and failure taxonomy draw on Cemri and colleagues’ [MAST study](https://arxiv.org/abs/2503.13657) and [released materials](https://github.com/multi-agent-systems-failure-taxonomy/MAST/tree/a70542e541b2104ef8fcd785778179e173fb8d70).
