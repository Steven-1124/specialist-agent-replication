# Rerun the experiment

Each of five surrounding teams performs the same 30 ProgramDev tasks with a generalist or specialist programmer. Each group–task–arm combination has three independent executions at temperature 1, for 900 executions. The six role prompts are identical across the two arms. The three evaluator channels measure completion, all 14 MAST modes, and P2 issues with trace locations. The completion votes are shared across the standard and P2 specifications.

## Reconstruct the historical evaluator inputs offline

Run these commands from the repository root with Python 3.10 or newer on Linux or macOS. The driver and replay use the Python standard library.

```sh
python3 experiment/run.py plan
python3 experiment/check_driver.py
python3 experiment/run.py replay
```

Replay feeds the archived outputs through the workflow, reconstructs the exact formatted JSON trace presented to the evaluators, and checks its SHA-256 digest. The released archive permits exact reconstruction for 657 executions. It marks 243 executions with redacted turns, which replay skips. During preparation, the same workflow reproduced the original judge input byte for byte for all 900 unredacted local executions, and its outgoing message lists matched all 7,512 recorded agent calls. The pass-level labels and historical analysis cover all 900 executions. The fixture test exercises routing, all three judgment parsers, repeated invocation, and majority voting without a model connection.

To save verified historical inputs for a new evaluation, run `python3 experiment/run.py replay --write-replay --out replayed`. Use a separate directory for newly generated runs.

## Generate new traces

[models.json](models.json) identifies each group's model, the two programmer models, the checkpoint revisions and loaders, and the endpoint environment variable for each model. The generalist programmer uses Qwen2.5-32B-Instruct and the specialist uses Qwen2.5-Coder-32B-Instruct. The five surrounding teams use Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct, Qwen3-Next-80B-A3B-Instruct, GPT-OSS-120B, and Llama-3.1-70B-Instruct, respectively. These are the study registry's revision pins. Historical service records establish the revisions for accepted executions of each model family. They do not establish a per-call weight hash for all 900 executions.

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

To execute a whole arm, omit the task and repeat filters. To execute all 900 runs, configure all seven model endpoints and remove all four filters. Hardware can be reused by starting only the surrounding model and coder for one group and arm at a time. Alternatively, set `MAS_BASE_URL` to a compatible model router and `MAS_API_KEY` if it requires authentication. Alternative backends can change decoding and numerical results. The original service used stochastic sampling without a fixed generation seed, top-p .95, and requested up to 8,192 output tokens per role. Two accepted specialist chess executions in G4 used a 16,384-token programmer allowance. [protocol.csv](protocol.csv) carries those exceptions for individual executions.

The portable driver implements the single fixed graph in [workflow.yaml](workflow.yaml). It reads the [six role templates](../prompts/README.md), uses the original ChatDev prompt-variable code, and passes only the designated messages and saved role state. An approval activates the manual stage. After three revision rounds, the graph can force a manual stage with an empty carried user message. Its system prompt still contains the available final code and feedback. No agent has access to external tools. The controller records messages and generated artifact contents as text, including code and manuals in the programmer and manual responses. It does not execute generated software.

## Obtain new judgments and summarize the new panel

Set `JUDGE_BASE_URL` to the DeepSeek chat-completions API base (the portion before `/chat/completions`) and `JUDGE_API_KEY` in your environment. The recorded alias is `deepseek-flash`. The optional `JUDGE_MODEL` override allows an explicitly labeled new evaluator. Evaluating one trace with all three channels requests nine model completions, and evaluating 900 traces requests 8,100. Compute time, account access, context limits, and API charges depend on the chosen services. Running the historical statistical analysis requires none of these calls.

```sh
python3 experiment/run.py judge --group 1 --arm generalist --task checkers --repeat 1 --out regenerated
python3 experiment/summarize.py --runs regenerated --tables new-tables
```

Remove the filters after generating the full sample. Use `--channel completion`, `--channel standard`, or `--channel p2` to run one channel. Each valid pass is saved separately. With the same inputs, rerunning resumes from those receipts. The driver stops on missing, malformed, or incompatible responses. It provides an explicit retry by invoking the command again. Historical recovery and retry selection were operational processes. New failures must be reported with their coverage.

Completion and standard failure modes are evaluated in independent requests. Within each of the 14 modes, as for completion, two of three votes determine the binary run result. P2 retains the issue descriptions and cited locations for subsequent review. The paper's frozen issue-coding decisions and feature ratings apply to the historical observations. Repeating the mechanism study with new assignments requires adjudicating the new P2 candidates and inspecting the new traces using the supplied [coding materials](../coding/).

The original DeepSeek evaluator regime applies to 785 runs (8,192-token maximum, temperature 1, top-p .95, thinking disabled, recorded reasoning effort none). The 115 continuation runs requested 262,144 tokens with thinking disabled and reasoning effort high. The original and continuation records use the same alias. Of the continuations, 108 have the later release marker and seven occurred on the upgrade date with an unpinned backend. Requested top-p may differ from effective nonthinking top-p. The driver preserves the request settings per run. A current service can change the model behind an alias or reject historical settings.

The summary command writes a CSV with the historical execution schema and records coverage, judge model, and whether the traces were freshly generated or archived and rejudged. The command requires three valid completion and three valid standard-mode responses per exported run. With the complete panel it uses the paper's estimator to compute arm means, 150 matched case states and risk frequency for each task across its five groups. Install [analysis/requirements.txt](../analysis/requirements.txt) before full-panel summarization. Use a new tables directory for each export. The historical analysis and its issue associations stay separate from new samples.

## Input and implementation provenance

[tasks.csv](tasks.csv) copies the 30 task descriptions in the original frozen study configuration, in their original order. These are the ProgramDev software-development tasks from Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* [MAST project](https://github.com/multi-agent-systems-failure-taxonomy/MAST) · [MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data)

The research implementation uses the frozen ChatDev workflow, the original placeholder-update routine, and the response and generated-artifact validation functions from the experiment. [ChatDev](https://github.com/OpenBMB/ChatDev) is distributed under its [Apache 2.0 license](ChatDev-LICENSE). The model service is the study's Hugging Face server with its import path adapted for standalone use. The portable controller replaces cluster scheduling and local credential handling with explicit commands, per-run directories, and environment variables. Server setup was checked offline. Fresh GPU generations and paid API calls were not part of release verification.
