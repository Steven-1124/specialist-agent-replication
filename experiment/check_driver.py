"""Offline integration check. All model and evaluator replies in this test are fixtures."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import run
from summarize import checked_votes


def main():
    tasks, rows = run.read_inputs()
    row = rows[0]
    counts = {"generation": 0, "evaluation": 0}
    modes = dict.fromkeys(run.MODE_IDS, 0)
    modes["1.1"] = 1
    code = {"phase": "Coding", "files": [{"file_name": "main.py", "code": "print('fixture')"}]}
    manual = {"phase": "Manual", "manual_file": "manual.md", "manual_content": "Fixture only."}
    agent_answers = iter([json.dumps({"phase": phase}) for phase in ("KPIDefinition", "ProductSpec", "TechnologyPlan")]
                         + [json.dumps(code), "<REVISE>\n{}", json.dumps(code), "<APPROVE>\n{}", json.dumps(manual)])

    def fixture(endpoint, model, options, token, timeout):
        if "response_format" not in options:
            counts["generation"] += 1
            assert options["max_tokens"] == 8192 and options["temperature"] == 1
            content = next(agent_answers)
        else:
            counts["evaluation"] += 1
            n = counts["evaluation"]
            assert options["thinking"] == {"type": "disabled"}
            assert options["max_tokens"] == 8192 and options["reasoning_effort"] == "none"
            payload = {"summary": "Fixture."}
            if n <= 3:
                payload["task_completion"] = n != 1
            else:
                payload["failure_modes"] = modes if n != 4 else dict.fromkeys(run.MODE_IDS, 0)
                if n > 6:
                    payload["failure_mode_locations"] = {"1.1": [{"issue": "Fixture defect.", "reference trace": "fixture"}]}
            content = json.dumps(payload)
        return {"model": model, "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}, content

    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        with patch.dict(os.environ, {"MAS_BASE_URL":"http://fixture.invalid", "JUDGE_BASE_URL":"http://fixture.invalid"}):
            with patch.object(run, "chat", fixture):
                run.generate(row, tasks[row["task"]], folder, 1)
                run.evaluate(row, folder, run.CHANNELS, 1)
                run.generate(row, tasks[row["task"]], folder, 1)
                run.evaluate(row, folder, run.CHANNELS, 1)
        assert counts == {"generation":8, "evaluation":9}
        result, _ = checked_votes(folder, row)
        assert result["mode_1.1"] == 1 and result["completion"] == 1
        assert json.loads((folder / "p2_voted.json").read_text())["total_failures"] == 1
        alternate = dict(row, evaluator_regime="native262144_chat_Flash-1017")
        try:
            checked_votes(folder, alternate)
        except ValueError:
            pass
        else:
            raise AssertionError("Mixed regimes should have failed.")
    print("Passed the offline two-round MAS, three-channel MAST, resume and export checks.")


if __name__ == "__main__":
    main()
