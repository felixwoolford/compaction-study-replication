"""SWE1: compaction on real coding tasks (SWE-bench Verified), without Docker.

A feasibility study outside the pre-registration (docs/build/swe1.md). Per task:
`ceval swe-prepare SWE1` clones the repository at its base commit, builds a uv
venv as SWE-bench's own specs say, makes the setup commit SWE-bench makes,
snapshots repo + venv and verifies the environment (the dataset's tests fail
before the gold patch and pass after it). `ceval run SWE1` then gives the agent
(pi through the bridge, built-in tools on, cwd = the task repository) one
instruction with the issue text and lets pi's own auto-compaction happen
mid-run; the agent's diff is scored on a fresh restore of the snapshot.

Layout (all under runs/SWE1/, gitignored):
  .tools/swebench-venv/   SWE-bench's package (harness/scripts/swe_tool.py runs here)
  .tools/python/          uv-managed interpreters (UV_PYTHON_INSTALL_DIR)
  .tools/hf/              the dataset parquet at the pinned revision
  <instance>/instance.json  dataset row + SWE-bench spec (harness-only: holds the gold patch)
  <instance>/repo/        the working copy the agent edits (reset from the snapshot)
  <instance>/venv/        the task's Python environment (on the agent's PATH)
  <instance>/pristine.tar repo + venv after setup (restored before every run and score)
  <instance>/prepared.json  timings, disk use and verification results
  <instance>/logs/        prepare and scoring logs
  <arm>/<instance>/wd/.pi, .lines/<arm>/<instance>/  the usual per-arm machinery (driver.py)

Outputs: results/SWE1/runs.jsonl (one row per (task, arm), append-only, resumable)
and results/SWE1/patches/<instance>__<arm>.diff (the agent's patch).
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import driver as D
from . import pisession as S
from . import schedule
from . import usageproxy
from .config import Config

# -----------------------------------------------------------------------------
# Paths and settings
# -----------------------------------------------------------------------------


def root_dir(cfg: Config, sc: dict) -> Path:
    return cfg.path(cfg["run"]["runs_dir"]) / sc["experiment"]


def task_dir(cfg: Config, sc: dict, iid: str) -> Path:
    return root_dir(cfg, sc) / iid


def tools_dir(cfg: Config, sc: dict) -> Path:
    return root_dir(cfg, sc) / ".tools"


def python_dir(cfg: Config, sc: dict) -> Path:
    return tools_dir(cfg, sc) / "python"


def swe_config(cfg: Config, sc: dict) -> Config:
    """The study config with this scenario's settings (a copy; Study 1's config
    and models file are untouched): the study-local models file (smaller
    declared window), pi's compaction settings for these arms, the per-run
    wall-clock cap and pi-fold's options."""
    s = sc["settings"]
    raw = copy.deepcopy(cfg.raw)
    raw["pi"]["models_file"] = s["models_file"]
    raw["pi"]["keep_recent_tokens"] = int(s["keep_recent_tokens"])
    raw["pi"]["reserve_tokens"] = int(s["reserve_tokens"])
    raw["model"]["context_window"] = int(s["context_window"])
    raw["bridge"]["turn_timeout_s"] = int(s["run_timeout_s"])
    if "fold_options" in s:
        raw["fold"]["options"] = s["fold_options"]
    return Config(raw=raw, root=cfg.root)


def check_settings(cfg: Config, sc: dict) -> list[str]:
    """Problems with the derived config: the models file must declare the
    study model with the scenario's window, an output limit below it, and the
    study's sampling; pi must compact below the window."""
    s = sc["settings"]
    out = []
    path = cfg.path(cfg["pi"]["models_file"])
    try:
        models = json.loads(path.read_text(encoding="utf-8"))
        entry = next(m for m in models["providers"][cfg["model"]["provider"]]["models"]
                     if m.get("id") == cfg["model"]["id"])
    except (OSError, ValueError, KeyError, StopIteration) as e:
        return [f"{path}: cannot find {cfg['model']['provider']}/{cfg['model']['id']} ({e})"]
    if int(entry.get("contextWindow", 0)) != int(s["context_window"]):
        out.append(f"{path}: contextWindow {entry.get('contextWindow')} != settings.context_window {s['context_window']}")
    if int(entry.get("maxTokens", 0)) >= int(s["context_window"]):
        out.append(f"{path}: maxTokens {entry.get('maxTokens')} is not below the window")
    want = cfg["model"].get("sampling_params")
    if want is not None and entry.get("samplingParams") != want:
        out.append(f"{path}: samplingParams {entry.get('samplingParams')} != config {want}")
    if int(s["reserve_tokens"]) + int(s["keep_recent_tokens"]) >= int(s["context_window"]):
        out.append("reserve_tokens + keep_recent_tokens must be below the window")
    return out


def check_arms(cfg: Config, arms: list[str]) -> None:
    """SWE1 has no A0: every arm must reduce context (pi compaction, any
    compactor behind it, or pi-fold's own folding)."""
    for a in arms:
        if a not in cfg["arms"]:
            raise ValueError(f"unknown arm {a!r}")
        if not cfg["arms"][a]["compact"] and not cfg["arms"][a].get("commit_turn"):
            raise ValueError(f"arm {a!r} never reduces context (A0); not part of SWE1")


def _uv() -> str:
    for c in (shutil.which("uv"), "/root/.local/bin/uv"):
        if c and Path(c).exists():
            return c
    raise RuntimeError("uv not found")


def _sh(cmd: str | list[str], cwd: Path, env: dict | None = None, timeout: int = 1800,
        log: Path | None = None) -> tuple[int, str]:
    """Run a command; returns (exit code, combined output). Timeout: exit 124."""
    shell = isinstance(cmd, str)
    try:
        p = subprocess.run(["bash", "-c", cmd] if shell else cmd, cwd=cwd, env=env, capture_output=True,
                           text=True, timeout=timeout, errors="replace")
        rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = 124
        out = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + \
            f"\n[timeout after {timeout}s]"
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n$ {cmd if shell else ' '.join(cmd)}\n[exit {rc}]\n{out}\n")
    return rc, out


GIT_ID = ["-c", "user.name=SWE-bench", "-c", "user.email=setup@swebench.config", "-c", "commit.gpgsign=false"]


def _git(repo: Path, *args: str, log: Path | None = None, timeout: int = 600) -> tuple[int, str]:
    return _sh(["git", *GIT_ID, *args], repo, env=_git_env(repo), timeout=timeout, log=log)


def _git_env(repo: Path) -> dict:
    env = dict(os.environ)
    env["GIT_CEILING_DIRECTORIES"] = str(repo.parent)
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return env


def task_env(td: Path, base: dict | None = None) -> dict:
    """The task's Python environment for install, test and the agent's shell:
    its venv first on PATH, no other venv, no PYTHONPATH, git confined to the
    task directory."""
    env = dict(os.environ if base is None else base)
    for k in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONSTARTUP", "FORCE_COLOR",
              "CLICOLOR_FORCE"):
        env.pop(k, None)
    # Plain test output, as in SWE-bench's containers: the log parsers match
    # lines starting with PASSED/FAILED, which colour codes would break.
    env["NO_COLOR"] = "1"
    env["PY_COLORS"] = "0"
    venv = td / "venv"
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["GIT_CEILING_DIRECTORIES"] = str(td)
    return env


def du(path: Path) -> int:
    """Bytes on disk under a path (files, not following symlinks)."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


# -----------------------------------------------------------------------------
# SWE-bench tool venv and dataset
# -----------------------------------------------------------------------------


def tool_python(cfg: Config, sc: dict) -> Path:
    return tools_dir(cfg, sc) / "swebench-venv" / "bin" / "python"


def ensure_tools(cfg: Config, sc: dict, log=print) -> Path:
    """The SWE-bench tool venv (separate from ceval; Python 3.12)."""
    py = tool_python(cfg, sc)
    if py.exists():
        return py
    t = sc["swebench"]
    venv = py.parent.parent
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(python_dir(cfg, sc)))
    env.pop("VIRTUAL_ENV", None)
    log(f"creating the SWE-bench tool venv at {venv}")
    for cmd in ([_uv(), "venv", "-q", "--python", t["tool_python"], str(venv)],
                [_uv(), "pip", "install", "-q", "--python", str(py), *t["tool_packages"]]):
        rc, out = _sh(cmd, root_dir(cfg, sc), env=env, timeout=900)
        if rc:
            raise RuntimeError(f"tool venv: {' '.join(cmd)} failed:\n{out[-2000:]}")
    return py


def _tool(cfg: Config, sc: dict, *args: str, timeout: int = 900) -> str:
    """Run swe_tool.py in the tool venv; returns its stdout (stderr carries warnings)."""
    script = cfg.root / "harness" / "scripts" / "swe_tool.py"
    p = subprocess.run([str(ensure_tools(cfg, sc)), "-I", str(script), *args], cwd=root_dir(cfg, sc),
                       capture_output=True, text=True, timeout=timeout, errors="replace")
    if p.returncode:
        raise RuntimeError(f"swe_tool {args[0]} failed:\n{(p.stdout + p.stderr)[-3000:]}")
    return p.stdout


def dataset_path(cfg: Config, sc: dict) -> Path:
    d = sc["dataset"]
    out = _tool(cfg, sc, "fetch", "--cache", str(tools_dir(cfg, sc) / "hf"), "--repo-id", d["name"],
                "--revision", d["revision"], "--sha256", d.get("sha256", ""))
    return Path(out.strip().splitlines()[-1])


def candidates(cfg: Config, sc: dict) -> dict[str, list[str]]:
    sel = sc["selection"]
    out = _tool(cfg, sc, "candidates", "--parquet", str(dataset_path(cfg, sc)), "--repos", ",".join(sel["repos"]),
                "--buckets", "|".join(sel["quota"]), "--seed", str(sel["seed"]))
    return json.loads(out[out.index("{"):])


def export_instances(cfg: Config, sc: dict, ids: list[str]) -> None:
    todo = [i for i in ids if not (task_dir(cfg, sc, i) / "instance.json").exists()]
    if todo:
        _tool(cfg, sc, "export", "--parquet", str(dataset_path(cfg, sc)), "--ids", ",".join(todo),
              "--out-dir", str(root_dir(cfg, sc)))


def load_instance(td: Path) -> dict:
    return json.loads((td / "instance.json").read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# SWE-bench log parsers (ports of swebench 4.1.0 harness/log_parsers/python.py;
# swe-prepare cross-checks them against the package on every verification log)
# -----------------------------------------------------------------------------

STATUSES = ("FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL")


def parse_log_pytest(log: str) -> dict[str, str]:
    out = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) <= 1:
                continue
            out[parts[1]] = parts[0]
    return out


def parse_log_pytest_options(log: str) -> dict[str, str]:
    option = re.compile(r"(.*?)\[(.*)\]")
    out = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) <= 1:
                continue
            m = option.search(parts[1])
            if m:
                main, opt = m.groups()
                if opt.startswith("/") and not opt.startswith("//") and "*" not in opt:
                    opt = "/" + opt.split("/")[-1]
                name = f"{main}[{opt}]"
            else:
                name = parts[1]
            out[name] = parts[0]
    return out


def parse_log_sympy(log: str) -> dict[str, str]:
    out = {}
    for m in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        out[f"{m[1]}.py:{m[2]}"] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                out[line.split()[0]] = "ERROR"
            if line.endswith(" F"):
                out[line.split()[0]] = "FAILED"
            if line.endswith(" ok"):
                out[line.split()[0]] = "PASSED"
    return out


PARSERS = {"parse_log_pytest": parse_log_pytest, "parse_log_pytest_options": parse_log_pytest_options,
           "parse_log_sympy": parse_log_sympy}


def grade(status: dict[str, str], f2p: list[str], p2p: list[str]) -> dict:
    """SWE-bench's grading (harness/grading.py, PASS_AND_FAIL): a test passes if
    PASSED or XFAIL, fails if FAILED, ERROR or absent from the log; SKIPPED
    counts as neither. Resolved: no FAIL_TO_PASS and no PASS_TO_PASS failure."""
    def outcome(t: str) -> str:
        s = status.get(t)
        if s in ("PASSED", "XFAIL"):
            return "pass"
        if s is None or s in ("FAILED", "ERROR"):
            return "fail"
        return "neither"
    per = {"FAIL_TO_PASS": {t: outcome(t) for t in f2p}, "PASS_TO_PASS": {t: outcome(t) for t in p2p}}
    c = {k: {o: sum(1 for v in d.values() if v == o) for o in ("pass", "fail", "neither")} for k, d in per.items()}
    return {"tests": per, "f2p_pass": c["FAIL_TO_PASS"]["pass"], "f2p_fail": c["FAIL_TO_PASS"]["fail"],
            "f2p_total": len(f2p), "p2p_pass": c["PASS_TO_PASS"]["pass"], "p2p_fail": c["PASS_TO_PASS"]["fail"],
            "p2p_total": len(p2p), "resolved": c["FAIL_TO_PASS"]["fail"] == 0 and c["PASS_TO_PASS"]["fail"] == 0}


# -----------------------------------------------------------------------------
# Snapshot, diff, scoring
# -----------------------------------------------------------------------------


def snapshot(td: Path) -> None:
    tmp = td / "pristine.tar.tmp"
    with tarfile.open(tmp, "w") as t:
        for name in ("repo", "venv"):
            t.add(td / name, arcname=name)
    tmp.replace(td / "pristine.tar")


def restore(td: Path) -> None:
    """Put repo and venv back exactly as after setup (whatever the agent or a
    scoring run did to either)."""
    for name in ("repo", "venv"):
        if (td / name).exists() or (td / name).is_symlink():
            shutil.rmtree(td / name)
    with tarfile.open(td / "pristine.tar") as t:
        t.extractall(td, filter="fully_trusted")  # our own archive (absolute venv symlinks)


def agent_diff(repo: Path, base: str) -> str:
    """The agent's change as one patch against the setup commit: tracked edits,
    new and deleted files (git add -A; .gitignore'd and excluded paths, such as
    .pi/ and setup artefacts, are left out)."""
    _git(repo, "add", "-A")
    rc, out = _sh(["git", "diff", "--cached", "--binary", base], repo, env=_git_env(repo))
    _git(repo, "reset", "-q")
    return out if rc == 0 else ""


def patch_files(patch: str) -> list[str]:
    return re.findall(r"^diff --git a/.* b/(.*)$", patch, re.M)


def score(cfg: Config, sc: dict, td: Path, inst: dict, patch: str, label: str,
          crosscheck: bool = False) -> dict:
    """SWE-bench's evaluation on a fresh restore: apply the model patch, reset
    the test_patch's files to base_commit, apply test_patch, rerun the install
    command, run the spec's test command on the test directives, parse and
    grade. Scoring problems (a patch that does not apply, a test timeout) are
    recorded, never retried. Leaves the task restored."""
    row, spec = inst["row"], inst["spec"]
    logs = td / "logs"
    log = logs / f"score-{label}.log"
    log.unlink(missing_ok=True)
    restore(td)
    repo = td / "repo"
    env = task_env(td)
    out: dict = {"scoring_error": None, "model_patch_applied": None, "test_patch_applied": None}
    t0 = time.monotonic()
    try:
        if patch.strip():
            pf = logs / f"score-{label}.model.diff"
            pf.write_text(patch, encoding="utf-8")
            rc, _ = _git(repo, "apply", "-v", "--whitespace=nowarn", str(pf), log=log)
            out["model_patch_applied"] = rc == 0
            if rc:
                out["scoring_error"] = "model patch does not apply"
                return out
        base = row["base_commit"]
        for f in inst["test_files"]:
            if _git(repo, "cat-file", "-e", f"{base}:{f}")[0] == 0:
                _git(repo, "checkout", base, "--", f, log=log)
            elif (repo / f).exists():
                (repo / f).unlink()  # new in test_patch: the agent's own file of that name goes
        tp = logs / f"score-{label}.test.diff"
        tp.write_text(row["test_patch"], encoding="utf-8")
        rc, _ = _git(repo, "apply", "-v", str(tp), log=log)
        out["test_patch_applied"] = rc == 0
        if rc:
            out["scoring_error"] = "test patch does not apply"
            return out
        if spec.get("install"):
            rc, _ = _sh(spec["install"], repo, env=env, timeout=1800, log=log)
            out["install_exit"] = rc
        cmd = " ".join([spec["test_cmd"], *inst["test_directives"]])
        rc, text = _sh(cmd, repo, env=env, timeout=int(sc["settings"]["test_timeout_s"]), log=log)
        out["test_exit"] = rc
        if rc == 124:
            out["scoring_error"] = "tests timed out"
        status = PARSERS[inst["parser"]](text)
        out.update(grade(status, row["FAIL_TO_PASS"], row["PASS_TO_PASS"]))
        if out["scoring_error"]:
            out["resolved"] = False
        if crosscheck:
            raw = logs / f"score-{label}.tests.txt"
            raw.write_text(text, encoding="utf-8")
            theirs = json.loads(_tool(cfg, sc, "parse", "--repo", row["repo"], "--log", str(raw)).strip().splitlines()[-1])
            out["parser_crosscheck"] = theirs == status
    finally:
        out["scoring_ms"] = int((time.monotonic() - t0) * 1000)
        restore(td)
    return out


# -----------------------------------------------------------------------------
# Preparing a task
# -----------------------------------------------------------------------------


def prepared(td: Path) -> dict | None:
    p = td / "prepared.json"
    if not p.exists() or not (td / "pristine.tar").exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _clone(td: Path, inst: dict, log: Path) -> None:
    """The repository at base_commit only (depth 1, no remote, no tags): the
    agent cannot reach later history, which holds the fix."""
    row = inst["row"]
    repo = td / "repo"
    repo.mkdir(parents=True)
    url = inst.get("clone_url") or f"https://github.com/{row['repo']}.git"
    for args in (["init", "-q", "-b", "main"], ["fetch", "-q", "--depth", "1", url, row["base_commit"]],
                 ["checkout", "-q", "FETCH_HEAD"], ["checkout", "-q", "-B", "main"]):
        rc, out = _git(repo, *args, log=log, timeout=1800)
        if rc:
            raise RuntimeError(f"git {' '.join(args)} failed: {out[-1000:]}")
    if _git(repo, "rev-parse", "HEAD")[1].strip() != row["base_commit"]:
        raise RuntimeError("HEAD is not base_commit")


def _make_venv(cfg: Config, sc: dict, td: Path, inst: dict, log: Path) -> str:
    """uv-managed CPython at the spec's version (no system interpreter), with
    pip/setuptools/wheel seeded: SWE-bench installs with `python -m pip`."""
    ver = inst["spec"]["python"]
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(python_dir(cfg, sc)), UV_PYTHON_PREFERENCE="only-managed")
    env.pop("VIRTUAL_ENV", None)
    for cmd in ([_uv(), "python", "install", ver], [_uv(), "venv", "-q", "--seed", "--python", ver, str(td / "venv")]):
        rc, out = _sh(cmd, td, env=env, timeout=900, log=log)
        if rc:
            raise RuntimeError(f"{' '.join(cmd)} failed: {out[-1000:]}")
    return _sh([str(td / "venv" / "bin" / "python"), "-V"], td)[1].strip()


def _install(td: Path, inst: dict, log: Path) -> None:
    """SWE-bench's environment and repository setup (test_spec/python.py
    make_env_script_list_py + make_repo_script_list_py), with pip in a venv
    instead of conda: requirements.txt from environment_setup_commit, or the
    spec's packages (conda package names, installed from PyPI), then
    pip_packages, the repo's install hook, pre_install and install."""
    spec, repo, env = inst["spec"], td / "repo", task_env(td)
    steps = []
    pkgs = spec.get("packages", "")
    if pkgs == "requirements.txt":
        (td / "requirements.txt").write_text(inst["requirements"] or "", encoding="utf-8")
        steps.append(f"python -m pip install -r {td / 'requirements.txt'}")
    elif pkgs == "environment.yml":
        raise RuntimeError("environment.yml specs need conda; not supported without Docker/conda")
    elif pkgs:
        steps.append(f"python -m pip install {pkgs}")
    if spec.get("pip_packages"):
        steps.append("python -m pip install " + " ".join(f"'{p}'" for p in spec["pip_packages"]))
    if inst.get("repo_install"):
        steps.append(inst["repo_install"])
    steps += list(spec.get("pre_install") or [])
    if spec.get("install"):
        steps.append(spec["install"])
    for s in steps:
        rc, out = _sh(s, repo, env=env, timeout=1800, log=log)
        if rc:
            raise RuntimeError(f"`{s}` failed (exit {rc}): {out[-1500:]}")


def _setup_commit(td: Path, log: Path) -> tuple[str, list[str]]:
    """SWE-bench's `git commit --allow-empty -am SWE-bench` after setup, so the
    agent's diff holds only its own changes. Untracked files setup left (and
    the arm's .pi/ directory) go to .git/info/exclude: never part of a diff."""
    repo = td / "repo"
    rc, out = _git(repo, "commit", "-q", "--allow-empty", "-am", "SWE-bench", log=log)
    if rc:
        raise RuntimeError(f"setup commit failed: {out[-1000:]}")
    untracked = [p for p in _sh(["git", "ls-files", "--others", "--exclude-standard"], repo,
                                env=_git_env(repo))[1].splitlines() if p]
    with (repo / ".git" / "info" / "exclude").open("a", encoding="utf-8") as f:
        f.write("\n# SWE1: the arm's pi project files and setup artefacts\n/.pi/\n")
        for p in untracked:
            f.write("/" + p + "\n")
    return _git(repo, "rev-parse", "HEAD")[1].strip(), untracked


def verify(cfg: Config, sc: dict, td: Path, inst: dict) -> dict:
    """The environment reproduces the dataset: with test_patch and no fix,
    every FAIL_TO_PASS test fails and no PASS_TO_PASS test fails; with the gold
    patch the task is resolved."""
    before = score(cfg, sc, td, inst, "", "verify-before", crosscheck=True)
    gold = score(cfg, sc, td, inst, inst["row"]["patch"], "verify-gold", crosscheck=True)
    ok_before = before.get("scoring_error") is None and before.get("f2p_pass") == 0 and before.get("p2p_fail") == 0
    ok_gold = gold.get("scoring_error") is None and bool(gold.get("resolved"))
    slim = lambda r: {k: v for k, v in r.items() if k != "tests"}  # noqa: E731
    return {"ok": ok_before and ok_gold and before.get("parser_crosscheck") is not False
            and gold.get("parser_crosscheck") is not False,
            "before": slim(before), "gold": slim(gold),
            "before_failing": {k: [t for t, o in d.items() if o != "pass"] for k, d in before.get("tests", {}).items()},
            "gold_failing": {k: [t for t, o in d.items() if o != "pass"] for k, d in gold.get("tests", {}).items()}}


def prepare_task(cfg: Config, sc: dict, iid: str, force: bool = False, log=print) -> dict:
    """Clone, build, set up, snapshot and verify one task. Idempotent: a task
    with a prepared.json and snapshot is left alone unless force."""
    td = task_dir(cfg, sc, iid)
    have = prepared(td)
    if have is not None and not force:
        return have
    inst = load_instance(td)
    for name in ("repo", "venv", "pristine.tar", "prepared.json", "logs"):
        p = td / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    plog = td / "logs" / "prepare.log"
    row = inst["row"]
    rec: dict = {"instance_id": iid, "repo": row["repo"], "version": row["version"],
                 "difficulty": row.get("difficulty"), "base_commit": row["base_commit"],
                 "python": inst["spec"]["python"], "swebench_version": inst.get("swebench_version"),
                 "ok": False, "error": None, "ts": datetime.now(timezone.utc).isoformat()}
    t0 = time.monotonic()
    try:
        _clone(td, inst, plog)
        rec["clone_s"] = round(time.monotonic() - t0, 1)
        t1 = time.monotonic()
        rec["python_exact"] = _make_venv(cfg, sc, td, inst, plog)
        _install(td, inst, plog)
        rec["env_build_s"] = round(time.monotonic() - t1, 1)
        rec["setup_commit"], rec["setup_untracked"] = _setup_commit(td, plog)
        snapshot(td)
        t2 = time.monotonic()
        rec["verification"] = verify(cfg, sc, td, inst)
        rec["verify_s"] = round(time.monotonic() - t2, 1)
        rec["ok"] = rec["verification"]["ok"]
        if not rec["ok"]:
            rec["error"] = "verification failed"
    except Exception as e:  # recorded: the task is replaced, never silently dropped
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["total_s"] = round(time.monotonic() - t0, 1)
    rec["disk_bytes"] = {"repo": du(td / "repo"), "venv": du(td / "venv"), "snapshot": du(td / "pristine.tar")}
    (td / "prepared.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if not rec["ok"]:
        (td / "pristine.tar").unlink(missing_ok=True)
    log(f"{iid}: {'ok' if rec['ok'] else 'FAILED: ' + str(rec['error'])} ({rec['total_s']}s)")
    return rec


def prepare(cfg: Config, sc: dict, ids: list[str] | None = None, force: bool = False, log=print) -> list[dict]:
    """Prepare the scenario's tasks (or `ids`); the results are also appended
    to results/<exp>/prepare.jsonl."""
    ids = ids or list(sc["tasks"])
    root_dir(cfg, sc).mkdir(parents=True, exist_ok=True)
    export_instances(cfg, sc, ids)
    out = []
    for iid in ids:
        fresh = force or prepared(task_dir(cfg, sc, iid)) is None
        rec = prepare_task(cfg, sc, iid, force=force, log=log)
        if fresh:
            _append(D.results_dir(cfg, sc["experiment"]) / "prepare.jsonl", rec)
        out.append(rec)
    return out


def select(cfg: Config, sc: dict, log=print) -> dict[str, list[str]]:
    """Walk the seeded candidate order per difficulty bucket and prepare
    candidates until each bucket's quota of verified tasks is met. Returns the
    chosen ids per bucket (to copy into the scenario's `tasks`); every attempt,
    failed ones included, is in results/<exp>/prepare.jsonl."""
    order = candidates(cfg, sc)
    chosen: dict[str, list[str]] = {}
    for bucket, n in sc["selection"]["quota"].items():
        chosen[bucket] = []
        for iid in order[bucket]:
            if len(chosen[bucket]) >= n:
                break
            rec = prepare(cfg, sc, [iid], log=log)[0]
            if rec["ok"]:
                chosen[bucket].append(iid)
    return chosen


# -----------------------------------------------------------------------------
# Running the agent
# -----------------------------------------------------------------------------


def _append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


_DONE = re.compile(r"\bDONE\b")
_NET = re.compile(r"\b(curl|wget|git\s+(fetch|pull|clone|remote\s+add)|pip3?\s+(install|download)|"
                  r"uv\s+(pip|add|sync)|python[\d.]*\s+-m\s+pip\s+(install|download))\b|https?://")


def network_commands(calls: list[dict]) -> list[str]:
    """Shell commands that may reach the network (fetch code, install packages):
    recorded per run, not an exclusion."""
    return [str((c.get("args") or {}).get("command", ""))[:300] for c in calls
            if c.get("name") == "bash" and _NET.search(str((c.get("args") or {}).get("command", "")))]


def session_env_reads(calls: list[dict]) -> list[str]:
    """Shell commands that name pi's session-file variable: pi's bash tool
    exports PI_SESSION_FILE (the live session, with everything compaction
    removed). The path-based audit cannot see a variable."""
    return [str((c.get("args") or {}).get("command", ""))[:300] for c in calls
            if c.get("name") == "bash" and "PI_SESSION" in str((c.get("args") or {}).get("command", ""))]


def compaction_rows(entries: list[dict], expected: str | None) -> list[dict]:
    out = []
    for e in entries:
        if e.get("type") != "compaction":
            continue
        who = S.compactor_of(e)
        out.append({"id": e.get("id"), "summary_chars": len(e.get("summary") or ""),
                    "tokens_before": e.get("tokensBefore"), "first_kept_id": e.get("firstKeptEntryId"),
                    "details_kind": (e.get("details") or {}).get("kind"), "compactor": who,
                    "treatment_ok": None if expected is None else who == expected,
                    "mech_checkpoint": S.mech_checkpoint_stats(entries, e) if who == "mech" else None})
    return out


class SWERunner:
    def __init__(self, cfg: Config, scenario: dict):
        from .runner import _git_commit
        self.sc = scenario
        self.exp = scenario["experiment"]
        self.cfg = swe_config(cfg, scenario)
        self.res_dir = D.results_dir(cfg, self.exp)
        self.rows_path = self.res_dir / "runs.jsonl"
        self.prompt_tpl = cfg.path(scenario["prompt"]).read_text(encoding="utf-8")
        self.meta = {"git_commit": _git_commit(cfg.root), "pi_version": cfg["pi"]["version"],
                     "model": f"{cfg['model']['provider']}/{cfg['model']['id']}"}
        self.done = {(r["instance_id"], r["arm"]) for r in _read_jsonl(self.rows_path)}

    def run(self, arms: list[str] | None = None, indices: list[int] | None = None, log=print) -> bool:
        """Run every (task, arm) without a row. False if stopped by the schedule
        gate (a started run finishes) or by a settings problem."""
        arms = arms or list(self.sc["arms"])
        check_arms(self.cfg, arms)
        problems = check_settings(self.cfg, self.sc)
        for p in problems:
            log(f"settings: {p}")
        if problems:
            return False
        tasks = list(self.sc["tasks"])
        if indices is not None:
            tasks = [tasks[i] for i in indices]
        proxy = None
        if self.cfg.raw.get("usage_proxy", {}).get("enabled"):
            proxy = usageproxy.UsageProxy(self.cfg["usage_proxy"]["upstream"]).start()
            D.set_usage_proxy(self.cfg, proxy, root_dir(self.cfg, self.sc) / ".usage-proxy-models.json")
        try:
            for iid in tasks:
                for arm in arms:
                    if (iid, arm) in self.done:
                        continue
                    ok, why = schedule.llm_allowed(self.cfg)
                    if not ok:
                        log(f"stopping before the next run: {why}")
                        return False
                    self.run_one(iid, arm, log)
                    self.done.add((iid, arm))
        finally:
            if proxy is not None:
                D.set_usage_proxy(self.cfg, None)
                proxy.stop()
        return True

    def run_one(self, iid: str, arm: str, log=print) -> dict | None:
        cfg, sc = self.cfg, self.sc
        td = task_dir(cfg, sc, iid)
        prep = prepared(td)
        if prep is None or not prep.get("ok"):
            log(f"{iid}/{arm}: skipped, task not prepared (run `ceval swe-prepare {self.exp}`)")
            return None
        inst = load_instance(td)
        row = inst["row"]
        restore(td)
        repo = td / "repo"
        wd = D.setup_workdir(cfg, self.exp, arm, iid)
        # pi reads project settings (and the lookup skill) from its cwd: the
        # arm's .pi goes into the repo, excluded from git (_setup_commit).
        shutil.copytree(wd / ".pi", repo / ".pi", dirs_exist_ok=True)
        env = D.arm_env(cfg, arm, self.exp, iid, overrides=sc.get("arm_env", {}).get(arm))
        env = task_env(td, env)
        session = D.sessions_dir(cfg, self.exp, arm, iid) / "main.jsonl"
        S.SessionWriter(session, cwd=str(repo), version=int(cfg["pi"]["session_version"]),
                        provider=cfg["model"]["provider"], model=cfg["model"]["id"], seed=0,
                        thinking_level=cfg["model"]["thinking_level"])
        prompt = self.prompt_tpl.format(repo_path=str(repo), problem_statement=row["problem_statement"].strip())
        cap = int(sc["settings"]["run_timeout_s"])
        t0 = time.monotonic()
        turns = [D.turn(cfg, session, repo, env, prompt)]
        nudged = False
        said_done = bool(_DONE.search(turns[0].text))
        base = prep["setup_commit"]
        if turns[0].ok and not said_done and not agent_diff(repo, base).strip():
            left = cap - int(time.monotonic() - t0)
            if left > 60:
                c2 = Config(raw=copy.deepcopy(cfg.raw), root=cfg.root)
                c2.raw["bridge"]["turn_timeout_s"] = left
                turns.append(D.turn(c2, session, repo, env, sc["nudge"]))
                nudged = True
                said_done = bool(_DONE.search(turns[-1].text))
        agent_ms = int((time.monotonic() - t0) * 1000)
        patch = agent_diff(repo, base)
        pdir = self.res_dir / "patches"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{iid}__{arm}.diff").write_text(patch, encoding="utf-8")

        entries = S.read_entries(session)
        calls = S.tool_calls(entries)
        tin, tout = S.token_usage(entries)
        names = [c["name"] for c in calls]
        # Anything under the code repo or the runs tree (instance.json holds the
        # gold patch; the session file holds what compaction removed) outside
        # the task repo and its venv/interpreter.
        allowed = (td / "venv", python_dir(cfg, sc))
        hits = sorted(set(D.leak_paths(calls, repo, allowed))
                      | set(D.leak_paths(calls, repo, allowed, root=cfg.path(cfg["run"]["runs_dir"]))))
        env_reads = session_env_reads(calls)
        comps = compaction_rows(entries, D.expected_compactor(cfg, arm))
        fold = S.fold_summary(entries) if "pi-fold" in D.arm_extensions(cfg, arm) else None
        usage = None
        for t in turns:
            usage = usageproxy.merge(usage, t.usage)
        err = next((t.error for t in turns if t.error), None)
        test_files = set(inst["test_files"])
        touched = patch_files(patch)
        res = score(cfg, sc, td, inst, patch, f"{arm}")
        out = {
            "experiment": self.exp, "instance_id": iid, "repo": row["repo"], "version": row["version"],
            "difficulty": row.get("difficulty"), "arm": arm,
            "resolved": res.get("resolved"), "f2p_pass": res.get("f2p_pass"), "f2p_total": res.get("f2p_total"),
            "p2p_pass": res.get("p2p_pass"), "p2p_total": res.get("p2p_total"),
            "f2p_fail": res.get("f2p_fail"), "p2p_fail": res.get("p2p_fail"),
            "scoring_error": res.get("scoring_error"), "model_patch_applied": res.get("model_patch_applied"),
            "test_patch_applied": res.get("test_patch_applied"), "test_exit": res.get("test_exit"),
            "tests": res.get("tests"), "scoring_ms": res.get("scoring_ms"),
            "patch_chars": len(patch), "patch_files": touched, "patch_empty": not patch.strip(),
            "patch_touches_test_patch_files": sorted(test_files & set(touched)),
            "said_done": said_done, "nudged": nudged, "turns": len(turns),
            "turn_ok": all(t.ok for t in turns), "turn_error": err,
            "timeout": bool(err and "timeout" in err.lower()),
            "agent_ms": agent_ms, "wall_ms": agent_ms + int(res.get("scoring_ms") or 0),
            "n_compactions": len(comps), "compactions": comps,
            "treatment_ok": None if not comps else all(c["treatment_ok"] is not False for c in comps),
            "fold": fold, "tokens_in": tin, "tokens_out": tout, "usage": usage,
            "n_tool_calls": len(calls), "tool_counts": {n: names.count(n) for n in sorted(set(names))},
            "n_messages": len(S.messages(entries)), "final_text": turns[-1].text[-2000:],
            "leak": bool(hits or env_reads), "leak_paths": hits, "leak_session_env": env_reads,
            "network_commands": network_commands(calls),
            "settings": {k: v for k, v in sc["settings"].items() if k != "note"}, "arm_env": D.arm_env_record(env),
            "extensions": D.arm_extensions(cfg, arm), "setup_commit": base,
            "ts": datetime.now(timezone.utc).isoformat(), **self.meta,
        }
        _append(self.rows_path, out)
        log(f"{iid}/{arm}: resolved={out['resolved']} compactions={out['n_compactions']} "
            f"tool_calls={out['n_tool_calls']} agent={agent_ms / 60000:.1f} min"
            + (f" error={err}" if err else ""))
        return out
