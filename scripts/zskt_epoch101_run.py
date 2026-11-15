"""Collect epoch-101 ZSKT images and score them with the existing Table 2 SVM."""
import builtins
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = Path(os.environ.get("PYTHON", sys.executable))
TMP = Path(os.environ.get("RUN_DIR", HERE))
STATUS = TMP / "zskt_e101_t200_status.json"
LOG = TMP / "zskt_e101_t200.log"
_REAL_PRINT = builtins.print
CRASH_AFTER_OK = {3221225477, -1073741819}


def print(*args, **kwargs):
    text = " ".join(str(arg) for arg in args)
    with open(LOG, "ab", buffering=0) as handle:
        handle.write((text + "\n").encode("utf-8", errors="replace"))
    _REAL_PRINT(text, flush=True)


def set_phase(phase, **extra):
    payload = {"phase": phase, "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(STATUS)
    print(f"PHASE {phase}")


def run(script, args, ok_marker):
    cmd = [str(PY), str(script), *args]
    print("RUN", " ".join(cmd))
    env = os.environ.copy()
    env["TEMP"] = str(TMP)
    env["TMP"] = str(TMP)
    env["PYTHONUNBUFFERED"] = "1"
    start = LOG.stat().st_size if LOG.exists() else 0
    with open(LOG, "ab", buffering=0) as handle:
        completed = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode == 0:
        return
    tail = LOG.read_bytes()[start:]
    if completed.returncode in CRASH_AFTER_OK and ok_marker.encode("utf-8") in tail:
        print(f"CHILD_CRASH_AFTER_OK code={completed.returncode} script={script.name}")
        return
    print(f"CHILD_FAIL code={completed.returncode} script={script.name}")
    raise SystemExit(completed.returncode)


def main():
    set_phase("collect")
    run(HERE / "zskt_collect_epoch101.py", ["--replace"], "COLLECT_DONE")
    set_phase("score")
    run(HERE / "zskt_score_epoch101.py", [], "SCORE_DONE")
    set_phase("done")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            set_phase("failed", code=code)
        raise
    except Exception:
        print(traceback.format_exc())
        set_phase("failed")
        raise SystemExit(1)
