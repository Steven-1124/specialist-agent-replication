# Better Outcome, Worse Process? The Hidden Cost of Using a Specialist Agent in an Agentic Team

Replication materials for the working paper by Yanxuan (Steve) Wu and Tingting Nian.

Five teams of agents each performed the same 30 software-development tasks in a multi-agent system adapted from ChatDev. Each team performed each task three times with a generalist coder and three times with a specialist coder, keeping the other four roles fixed. Across the 900 runs, judged completion rose from **32.4% to 48.9%** and the mean number of distinct failure modes flagged per run fell from **2.79 to 2.55**.

## Results by task

For **35 of 150 team–task pairs (23.3%)**, completion held steady or rose with the specialist coder while more distinct failure modes were flagged. These *risk cases* spanned **21 of the 30 tasks**. The table reports each task’s averages across 15 runs per coder condition. The last column counts how many of the five teams entered the risk state on that task.

| Task | Judged completion, generalist → specialist | Flagged modes per run, generalist → specialist | Teams in the risk state |
| --- | ---: | ---: | ---: |
| Gold Miner — time a moving claw to collect gold before the level ends | 13.33% → 26.67% | 2.13 → 3.20 | 5 of 5 |
| The Crossword — enter words by clue and direction and check intersecting letters | 6.67% → 60.00% | 2.87 → 2.47 | 3 of 5 |
| Sudoku — fill a 9×9 grid, identify mistakes, and check the completed puzzle | 40.00% → 60.00% | 2.73 → 1.87 | 0 of 5 |

<details>
<summary>Show the other 27 tasks</summary>

| Task | Judged completion, generalist → specialist | Flagged modes per run, generalist → specialist | Teams in the risk state |
| --- | ---: | ---: | ---: |
| 2048 | 26.67% → 66.67% | 2.60 → 2.80 | 2 of 5 |
| Budget tracker | 0.00% → 13.33% | 3.60 → 3.00 | 0 of 5 |
| Candy Crush | 26.67% → 40.00% | 2.67 → 3.33 | 3 of 5 |
| Checkers | 6.67% → 26.67% | 2.87 → 2.53 | 1 of 5 |
| Chess | 0.00% → 6.67% | 3.93 → 3.40 | 1 of 5 |
| Connect Four | 80.00% → 86.67% | 1.80 → 1.87 | 1 of 5 |
| Connections (NYT) | 0.00% → 6.67% | 4.00 → 3.27 | 1 of 5 |
| Detect palindromes | 73.33% → 73.33% | 1.87 → 1.27 | 0 of 5 |
| Doudizhu Poker | 0.00% → 6.67% | 2.93 → 3.20 | 2 of 5 |
| Episode | 20.00% → 40.00% | 2.60 → 2.47 | 1 of 5 |
| Fibonacci numbers | 100.00% → 100.00% | 1.47 → 1.27 | 0 of 5 |
| Flappy Bird | 33.33% → 26.67% | 3.33 → 2.87 | 0 of 5 |
| Gomoku | 53.33% → 73.33% | 2.27 → 2.60 | 2 of 5 |
| Mastermind | 66.67% → 86.67% | 2.20 → 2.07 | 1 of 5 |
| Minesweeper | 20.00% → 46.67% | 2.80 → 2.33 | 2 of 5 |
| Monopoly Go | 13.33% → 13.33% | 3.93 → 3.53 | 0 of 5 |
| Pong | 66.67% → 86.67% | 2.47 → 2.27 | 1 of 5 |
| Reversi / Othello | 53.33% → 60.00% | 2.87 → 2.80 | 0 of 5 |
| Snake | 26.67% → 60.00% | 2.47 → 2.40 | 0 of 5 |
| Space Invaders | 33.33% → 66.67% | 2.80 → 3.00 | 2 of 5 |
| Strands (NYT) | 0.00% → 0.00% | 3.93 → 3.73 | 2 of 5 |
| Strands game | 20.00% → 26.67% | 3.40 → 2.87 | 1 of 5 |
| Tetris | 20.00% → 53.33% | 2.67 → 2.20 | 1 of 5 |
| Tic-tac-toe | 53.33% → 66.67% | 2.07 → 1.47 | 1 of 5 |
| Tiny Rouge | 13.33% → 46.67% | 3.33 → 2.27 | 0 of 5 |
| Trivia quiz | 53.33% → 66.67% | 2.80 → 2.20 | 1 of 5 |
| Wordle | 53.33% → 73.33% | 2.40 → 2.00 | 1 of 5 |

</details>

The Crossword illustrates why the team-level comparison matters. Across all five teams, completion rose from 6.67% to 60.00% and flagged modes fell from 2.87 to 2.47 per run. Yet three of those teams recorded more failure modes while maintaining or improving completion. Across the 30 tasks, risk cases were more frequent when requirements interacted or correctness depended on a sequence of actions. These task features were closely correlated and had limited predictive value for held-out tasks. [Task-feature estimates](data/published/feature_models.csv)

[Browse all 30 task results](data/task_results.csv) · [See each team's state on each task](data/published/task_risk_frequency.csv) · [Read the original task requests](experiment/tasks.csv)

## Which five failure modes accounted for 91.3% of the gap?

Section 5.1 compares the 35 risk cases with the 74 *benefit cases*, where completion also held steady or rose but fewer modes were flagged. For each mode, we count how many of the three runs were flagged with each coder. We subtract the generalist count from the specialist count within each team–task pair and average that change separately for risk and benefit cases. The final column subtracts the benefit mean from the risk mean. [All 14 mode contrasts](data/published/mode_contrasts.csv)

| MAST mode | Failure mode | Mean change in risk cases | Mean change in benefit cases | Risk minus benefit |
| --- | --- | ---: | ---: | ---: |
| 2.6 | Reasoning–action mismatch | +1.029 | −0.676 | +1.704 |
| 1.1 | Missed task requirements | +0.600 | −0.554 | +1.154 |
| 3.3 | Incorrect verification | +0.457 | −0.662 | +1.119 |
| 2.5 | Ignored input from teammates | +0.657 | −0.432 | +1.090 |
| 1.3 | Repeated steps | +0.343 | −0.338 | +0.681 |

Using unrounded estimates, the five largest contrasts sum to **5.747876** flagged occurrences per case. A run contributes once for each distinct mode flagged. The signed contrasts across all 14 modes sum to **6.293050**. Dividing the first sum by the second gives **91.3%** of the net gap in how process failures changed. [Code for the mode contrasts](analysis/reproduce.py#L55-L69) · [Code for the ranking and 91.3% calculation](analysis/reproduce.py#L72-L81)

## Reproduce the results

To reconstruct and print the 30 task rows from the saved evaluator judgments, clone the repository and run the verifier with Python 3.10 or later. This step uses only the standard library. [Task verification code](verify.py)

```sh
git clone https://github.com/Steven-1124/specialist-agent-replication.git
cd specialist-agent-replication
python3 verify.py
```

To reproduce the five-mode ranking and its share of the gap, run the following from the repository root. The last command checks the numerical tables against the archived results, writes them to `build/analysis/`, and prints the five mode IDs, their contrasts, and the calculation `5.747876 / 6.293050 = 91.3%`.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py --no-plots
```

To generate the paper’s ten empirical figures as well, run `python analysis/reproduce.py`. [Analysis guide and exhibit index](analysis/README.md)

The [experimental workflow](experiment/README.md) and [prompts](prompts/README.md) document how the teams produced the saved work. The workflow adapts [ChatDev](https://github.com/OpenBMB/ChatDev). The tasks and failure taxonomy draw on Cemri and colleagues’ [MAST study](https://arxiv.org/abs/2503.13657) and [released materials](https://github.com/multi-agent-systems-failure-taxonomy/MAST/tree/a70542e541b2104ef8fcd785778179e173fb8d70).
