# Archived agent traces and annotation evidence

Choose a task from the [task results](../data/task_results.csv) or [original task requests](../experiment/tasks.csv), then use its `task_id` to read the team’s work. For example, these commands verify the archive, show the first role’s output for a Checkers run, and display the passages supporting one of its issue annotations. Run them from the repository root with Python’s standard library.

```sh
python3 traces/verify.py
python3 traces/read.py G1 checkers generalist 1 --sequence 1
python3 traces/read.py G1 checkers generalist 1 --issues --label S_CEO
```

The viewer prints generated code as text. Each file is UTF-8 JSON Lines inside gzip, so a reader can also stream it with `gzip.open(path, "rt", encoding="utf-8")` and `json.loads(line)`.

## Run selection and recorded outputs

Each of the 30 tasks has three runs in each programmer condition for each of five groups. Use `group`, `task_id`, `arm` and `run_number` to join a trace to its [completion and failure-mode judgments](../data/executions.csv). Each trace retains the identifier and source hashes of the execution used in the paper.

Each row in [runs.jsonl.gz](runs.jsonl.gz) contains the public run key, historical run ID, full and simplified source-file hashes, and the ordered turns. Each turn contains its original sequence number, role name and generated output. There are 7,512 turns. Most inputs repeat preceding outputs and the common role instructions. The [prompts](../prompts/) give those instructions, and the issue crosswalk preserves the specific input passages used in annotations. Provider metadata, credentials, local directory names and timestamps are unnecessary for reading the role outputs.

In 311 turns across 243 runs, tildes mask contact addresses, credential-like literals, connection strings containing user information, external URLs, an example home path, or embedded image data. Public documentation links and local demonstration URLs may remain. The mask replaces each affected character with one tilde so that quoted passages stay at the same offsets. Each row's `masked_turns` field identifies these turns. The manifest records counts by masking category and the archive checksums. The source hashes refer to the historical files before this projection and masking.

## Following an issue annotation

The `association` field in [issue_anchors.jsonl.gz](issue_anchors.jsonl.gz) is the one-based **data-row number** of [issue_associations.csv](../data/issue_associations.csv), excluding its header. Each record also carries the run key, issue label, primary failure mode, role, object and event numbers, archived evidence status and recorded turn numbers. Missing object or event numbers appear as `null`. The 6,131 records are associations between an observed entry or rule occurrence and an issue type. One run can have multiple associations.

The original annotation files contain 24,688 stored anchors. All were located in the historical run. Of these, 21,388 point to a role output, 3,080 to a role input and 220 to the task request. A character span starts at `start` (zero-based) and stops before `end`, with offsets measured in Unicode code points. For outputs, slice the selected turn’s output at those offsets. For inputs and task requests, the archive supplies the source excerpt. For 546 passages the original output serialized the quoted text inside a JSON string. These records carry `representation: JSON-escaped`. When a quotation can occur more than once within the same field and turn, the offsets select its first occurrence.

For 82 anchors, the historical entry left the field unspecified and the quoted passage was found in the output. Two more named the output but the quoted text was found in the input. Such anchors preserve the original designation as `recorded_field`. Another 492 associations have no stored excerpt. These preserve their archived status and any recorded turn numbers. Readers can inspect the corresponding run and the issue rule to assess the classification. Locating the words in a trace checks the quotation. Interpreting the passage as a failure also requires applying the coding rule.

[manifest.json](manifest.json) records SHA-256 digests for the distributed gzip files. [verify.py](verify.py) checks these digests, the full set of execution keys, the correspondence with every published annotation row and the character bounds of the released passages.
