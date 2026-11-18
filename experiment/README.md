# Run a ProgramDev task

The [task requests](tasks.csv) contain the exact instructions for the 30 programs in the study. The [task results](../data/task_results.csv) show what happened under each programmer condition. For each task, five groups used both the generalist and specialist programmers, with three runs per group and condition. The programmer’s instructions and the surrounding team were held fixed within a group.

## Replay archived work

Run these commands from the repository root with Python 3.10 or newer on Linux or macOS. The controller uses the Python standard library.

```sh
python3 experiment/run.py plan
python3 experiment/check_driver.py
python3 experiment/run.py replay
```

Replay passes saved outputs through the workflow and checks the reconstructed evaluator input against its recorded SHA-256 digest. The public archive supports exact replay for 657 runs. Replay skips the 243 runs containing masked text. The saved judgments and [paper analysis](../analysis/README.md) cover all 900 runs. Use `replay --write-replay --out replayed` to save the inputs that pass the check.

## Generate new traces

The [model assignments](models.json) identify the noncoder model for each group and the model revision, loader, and endpoint variable for each agent. The generalist programmer uses Qwen2.5-32B-Instruct, and the specialist uses Qwen2.5-Coder-32B-Instruct. The archived revision pins apply to model families. Per-call weight hashes were not recorded.

The supplied [model server](serving/hf_chat_server.py) uses the checkpoint's native chat template and tokenizer. Qwen and Llama use bitsandbytes NF4. GPT-OSS uses native MXFP4 and Harmony. Obtain the checkpoint access and weights from each model's publisher. Each server process needs one visible CUDA GPU with sufficient memory for its model and the long-context workload. Install CUDA-compatible PyTorch in a Python 3.10 environment and the packages in [the base requirements](serving/requirements.txt). For Qwen3-Next and GPT-OSS, create a separate serving environment with the same base packages, then install [the modern overrides](serving/requirements-modern.txt). The controller runs independently of these GPU environments.

For a Group 1 generalist run, start the following processes in separate terminals on a model host with two suitable GPUs. Use the same host for the example controller, or forward its ports to the model host.

```sh
CUDA_VISIBLE_DEVICES=0 python experiment/serving/serve.py llama-3.3-70b-instruct --port 8101
CUDA_VISIBLE_DEVICES=1 python experiment/serving/serve.py qwen2.5-32b-instruct --port 8102
```

In the controller terminal, set the corresponding endpoints and run one planned execution.

```sh
export OS_MAS_LLAMA33_70B_BASE_URL=http://127.0.0.1:8101/v1
export OS_MAS_QWEN25_32B_BASE_URL=http://127.0.0.1:8102/v1
python3 experiment/run.py generate --group 1 --arm generalist --task checkers --repeat 1 --out regenerated
```

To run every task for one group and programmer condition, omit `--task` and `--repeat`. To run all five groups under both conditions, configure all seven endpoints and remove all four filters. The controller can also use a compatible router through `MAS_BASE_URL` and, if required, `MAS_API_KEY`. The [run configuration](run_schedule.csv) and [model assignments](models.json) preserve the settings used in the study. Generation used stochastic sampling without a fixed seed, so a new run can produce a different answer.

The [workflow](workflow.yaml) uses the [role templates](../prompts/README.md) to pass requirements, implementation plans, code, and review decisions between roles. After approval or the three-round limit, the CPO writes the manual from the saved program. The controller records the messages and artifacts as text. It does not execute the generated software.

## Judge and summarize newly generated tasks

Set `JUDGE_BASE_URL` to the evaluator service's chat API base URL (before `/chat/completions`) and `JUDGE_API_KEY` to its access key. The [driver](run.py) records the evaluator alias and applies the archived per-run settings. Access to the service is needed only for fresh judgments. The historical estimates can be reproduced entirely from the saved data.

```sh
python3 experiment/run.py judge --group 1 --arm generalist --task checkers --repeat 1 --out regenerated
python3 experiment/summarize.py --runs regenerated --tables new-tables
```

Use the same filters for generation and judging. Each run produces three judgments each for completion, failure modes, and issue identification. With the full sample generated and judged, install the [analysis dependencies](../analysis/requirements.txt) and summarize to obtain paired case states and task risk frequencies. The summary also records coverage. New traces require a new review of nominated issues under the study's [coding rules](../coding/issue_rules.csv).

## Input and implementation provenance

[tasks.csv](tasks.csv) copies the 30 task descriptions in the original frozen study configuration, in their original order. These are the ProgramDev software-development tasks from Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* [MAST project](https://github.com/multi-agent-systems-failure-taxonomy/MAST) · [MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data)

The research implementation uses the frozen ChatDev workflow, the original placeholder-update routine, and the response and generated-artifact validation functions from the experiment. [ChatDev](https://github.com/OpenBMB/ChatDev) is distributed under its [Apache 2.0 license](ChatDev-LICENSE). The model service is the study's Hugging Face server with its import path adapted for standalone use. The portable controller replaces cluster scheduling and local credential handling with explicit commands, per-run directories, and environment variables. Server setup was checked offline. Fresh GPU generations and paid API calls were not part of release verification.
