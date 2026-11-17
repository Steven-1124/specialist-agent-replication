"""Start the study model service using the model registry and an explicit GPU."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
models = json.loads((HERE.parent / "models.json").read_text())["models"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", choices=tuple(models))
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--record-dir", type=Path, default=Path("service-records"))
args = parser.parse_args()
m = models[args.model]
args.record_dir.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("OS_MAS_STRICT_CONTEXT_BUDGET", "1")
command = [sys.executable, str(HERE / "hf_chat_server.py"), "--model-key", args.model,
           "--model-id", m["id"], "--served-model-name", m["id"], "--revision", m["revision"],
           "--loader", m["loader"], "--endpoint-env", m["endpoint_env"], "--port", str(args.port),
           "--service-record", str(args.record_dir / (args.model + ".json"))]
os.execv(sys.executable, command)
