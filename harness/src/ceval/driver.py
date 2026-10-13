"""Talking to pi: per-arm working directories, the Node bridge, turns and compactions.

The bridge (harness/bridge/pi-bridge.mjs) only makes pi *do* something to a
session file. Everything we measure is read back from the session file here,
so the measurement code does not depend on how the bridge is implemented.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from . import pisession as S
from . import usageproxy
from . import schedule
from .config import Config, repo_root


# -----------------------------------------------------------------------------
# Working directories
# -----------------------------------------------------------------------------


def workdir(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """Per-arm, per-session base directory: the agent's project dir plus the
    harness's session files and compaction dumps."""
    return cfg.path(cfg["run"]["runs_dir"]) / exp / arm / session_id


def agent_cwd(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """The working directory the agent sees: an empty project dir."""
    return workdir(cfg, exp, arm, session_id) / "wd"


def results_dir(cfg: Config, exp: str) -> Path:
    """results/<exp>/, except second-pilot experiments (P1, P1B, P1C, P2, P3), which go to
    results/pilot2/<exp>/ (never part of the confirmatory analysis)."""
    base = cfg.path(cfg["run"]["results_dir"])
    return base / "pilot2" / exp if exp in ("P1", "P1B", "P1C", "P2", "P3") else base / exp


def lines_dir(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """Harness-owned files for one (arm, session): kept in runs/<exp>/.lines/,
    not next to the agent's working directory (PROTOCOL.md v2 §6.1). The agent
    runs as root and could still find them; the leak audit catches that."""
    return cfg.path(cfg["run"]["runs_dir"]) / exp / ".lines" / arm / session_id


def sessions_dir(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """Session files (main line + probe forks), away from the agent's cwd."""
    return lines_dir(cfg, exp, arm, session_id) / "sessions"


def dumps_dir(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """Compaction dumps (via MECH_COMPACT_DUMP_DIR), away from the agent's cwd;
    lookup arms reach them through context_lookup."""
    return lines_dir(cfg, exp, arm, session_id) / "dumps"


def home_dir(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """The bridge's private HOME for this line (every arm): pi-lcm and pi-fold
    read their settings from, and could write state under, the home directory;
    none of that may touch the real home. pi's own agent dir is still passed
    explicitly (CEVAL_PI_AGENT_DIR)."""
    return lines_dir(cfg, exp, arm, session_id) / "home"


def lcm_dir(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """pi-lcm's SQLite store for the main line (LCM_DB_DIR)."""
    return lines_dir(cfg, exp, arm, session_id) / "lcm"


def vcc_config_path(home: Path) -> Path:
    """pi-vcc's config file (PI_VCC_CONFIG_PATH) inside a private HOME. pi-vcc
    0.8.12 creates its config file at load (src/core/settings.ts
    scaffoldSettings), by default at getAgentDir()/pi-vcc-config.json; the
    harness points it into the line's private HOME so nothing is written to the
    real ~/.pi/agent. The harness writes no config: the package's defaults apply."""
    return home / ".pi-vcc" / "pi-vcc-config.json"


def probe_state_dir(cfg: Config, exp: str, arm: str, session_id: str, name: str) -> Path:
    """Per-probe copies of the line's out-of-session state (home, lcm)."""
    return lines_dir(cfg, exp, arm, session_id) / "probe-state" / name


def arm_extensions(cfg: Config, arm: str) -> list[str]:
    """Registry names of the extensions the arm loads (exactly these)."""
    names = cfg["arms"][arm].get("extensions")
    if names is None:
        raise ValueError(f"arm {arm!r} lists no extensions (use [] for none)")
    reg = cfg["extension"]["registry"]
    unknown = [n for n in names if n not in reg]
    if unknown:
        raise ValueError(f"arm {arm!r}: unknown extensions {unknown}")
    if cfg["arms"][arm]["skill"] and not {"mech", "verbatim"} & set(names):
        raise ValueError(f"arm {arm!r}: the recovery skill belongs to mech or verbatim")
    return list(names)


def _registry_source(cfg: Config, name: str) -> Path:
    src = str(cfg["extension"]["registry"][name]["source"])
    if src.startswith("@"):  # a key of the extension section, e.g. @mech_compact_ts
        src = str(cfg["extension"][src[1:]])
    return cfg.path(src).resolve()


def extension_paths(cfg: Config, arm: str, wd: Path) -> list[Path]:
    """Absolute paths the bridge loads for the arm, in the arm's order: 'copy'
    extensions from the arm's wd/.pi/extensions/, 'in-place' packages from
    their package directory (they need their own node_modules)."""
    out = []
    for n in arm_extensions(cfg, arm):
        e = cfg["extension"]["registry"][n]
        out.append((wd / ".pi" / "extensions" / _registry_source(cfg, n).name).resolve()
                   if e["install"] == "copy" else _registry_source(cfg, n))
    return out


def expected_compactor(cfg: Config, arm: str) -> str | None:
    """What must have produced the arm's compaction rows: 'pi' (pi's own
    summary), a registry compactor ('mech', 'lcm', 'fold', 'vcc'), or None (unchecked)."""
    return cfg["arms"][arm].get("compactor")


def compact_timeout_s(cfg: Config, arm: str) -> int:
    return int(cfg["arms"][arm].get("compact_timeout_s", cfg["bridge"]["compact_timeout_s"]))


def lcm_settings(cfg: Config) -> dict:
    """pi-lcm reads compactionModels only from $HOME/.pi/agent/settings.json
    (src/config.ts readSettingsLcm; no env var). Written into the private HOME
    so its summariser is the study model: the package default list (cerebras,
    anthropic) is replaced, and its fallback is the session model, also the
    study model (index.ts callCompactionModel)."""
    return {"lcm": {"compactionModels": [{"provider": cfg["model"]["provider"], "id": cfg["model"]["id"]}]}}


def setup_workdir(cfg: Config, exp: str, arm: str, session_id: str) -> Path:
    """Reset the per-arm work area; return the agent's working directory.

    Layout:
      runs/<exp>/<arm>/<sid>/wd/            agent cwd: copied extensions (mech),
                                            skill (lookup arms only), project
                                            .pi/settings.json
      runs/<exp>/.lines/<arm>/<sid>/sessions/  main.jsonl + forks (harness-owned)
      runs/<exp>/.lines/<arm>/<sid>/dumps/     compaction dumps (MECH_COMPACT_DUMP_DIR)
      runs/<exp>/.lines/<arm>/<sid>/home/      the bridge's private HOME
      runs/<exp>/.lines/<arm>/<sid>/lcm/       pi-lcm's store (pi-lcm arms)
      runs/<exp>/.lines/<arm>/<sid>/probe-state/<probe>/  per-probe copies of home, lcm
    """
    names = arm_extensions(cfg, arm)
    wd = agent_cwd(cfg, exp, arm, session_id)
    lines = lines_dir(cfg, exp, arm, session_id)
    fresh = [wd, sessions_dir(cfg, exp, arm, session_id), dumps_dir(cfg, exp, arm, session_id),
             home_dir(cfg, exp, arm, session_id), lines / "probe-state"]
    if "pi-lcm" in names:
        fresh.append(lcm_dir(cfg, exp, arm, session_id))
    elif lcm_dir(cfg, exp, arm, session_id).exists():
        shutil.rmtree(lcm_dir(cfg, exp, arm, session_id))
    for d in fresh:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    ext_dir = wd / ".pi" / "extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        if cfg["extension"]["registry"][n]["install"] == "copy":
            src = _registry_source(cfg, n)
            shutil.copyfile(src, ext_dir / src.name)
    (wd / ".pi" / "settings.json").write_text(json.dumps(project_settings(cfg, arm), indent=2) + "\n",
                                              encoding="utf-8")
    home = home_dir(cfg, exp, arm, session_id)
    if "pi-lcm" in names:
        _write_json(home / ".pi" / "agent" / "settings.json", lcm_settings(cfg))
    if "billion" in names:
        # billion-context-pi reads ~/.pi/acp.json (the private HOME): its own
        # sub-agent tools off, so it offers exactly its compression tools.
        _write_json(home / ".pi" / "acp.json", {"delegate": False})
    skill = cfg["arms"][arm]["skill"]
    if skill:  # true: the context-dump skill; a path: that arm's own skill
        src = cfg.path(skill if isinstance(skill, str) else cfg["extension"]["skill_dir"])
        # The folder is named after the skill's own `name:` (context-dump for mech's
        # skills, context-retrieval for the release's).
        name = None if skill is True else re.search(r"^name:\s*(\S+)", (src / "SKILL.md").read_text(encoding="utf-8"), re.M)
        shutil.copytree(src, wd / ".pi" / "skills" / (name.group(1) if name else "context-dump"))
    return wd


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def copy_probe_state(cfg: Config, exp: str, arm: str, session_id: str, name: str) -> dict[str, str]:
    """Copy the line's out-of-session state (private HOME; pi-lcm's store) for
    one probe and return the env overrides pointing the probe at the copy.
    pi-lcm stores every message it sees (message_end) in its store, keyed by pi
    session id; probe traffic must never reach the main line's store, and each
    probe must start from the state the round's compaction left. Called after
    the round's compaction, before the probe's first turn."""
    dst = probe_state_dir(cfg, exp, arm, session_id, name)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    env = {}
    shutil.copytree(home_dir(cfg, exp, arm, session_id), dst / "home", symlinks=True)
    env["HOME"] = str(dst / "home")
    if "pi-vcc" in arm_extensions(cfg, arm):
        env["PI_VCC_CONFIG_PATH"] = str(vcc_config_path(dst / "home"))
    if "pi-lcm" in arm_extensions(cfg, arm):
        # The bridge process that wrote the store has exited: the db, -wal and
        # -shm files are consistent on disk and are copied together.
        shutil.copytree(lcm_dir(cfg, exp, arm, session_id), dst / "lcm")
        env["LCM_DB_DIR"] = str(dst / "lcm")
    return env


def project_settings(cfg: Config, arm: str | None = None) -> dict:
    """pi project settings written into every arm's cwd. The study's compaction
    settings live here, not in the user's global settings file; pi merges
    project settings over global ones. Arms that never compact (A0) also get
    auto-compaction disabled, so a long context cannot silently turn A0 into
    pi-default (PROTOCOL.md v2 §6.1)."""
    comp = {"keepRecentTokens": int(cfg["pi"]["keep_recent_tokens"]),
            "reserveTokens": int(cfg["pi"]["reserve_tokens"])}
    if arm is not None and not cfg["arms"][arm]["compact"]:
        comp["enabled"] = False
    return {"compaction": comp}


def summary_max_tokens(cfg: Config) -> int:
    """pi's summariser budget: floor(0.8 * reserveTokens) (compaction.js
    generateSummaryWithUsage). The designed arm gets the same budget, so A2
    differs from A1 only in the prompt."""
    return int(0.8 * int(cfg["pi"]["reserve_tokens"]))


def arm_env(cfg: Config, arm: str, exp: str | None = None, session_id: str | None = None,
            overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The arm's environment. `overrides` are per-study settings from the
    scenario's `arm_env` (e.g. Study 3's larger mech cap), applied last."""
    m = cfg["model"]
    env = dict(os.environ)
    env.update({
        "PI_PACKAGE_DIR": str(cfg["pi"]["package_dir"]),
        "CEVAL_PI_AGENT_DIR": str(Path(cfg["pi"]["settings_file"]).expanduser().parent),
        "CEVAL_PI_MODELS_FILE": str(cfg.path(cfg["pi"]["models_file"])),
        "CEVAL_MODEL_PROVIDER": m["provider"],
        "CEVAL_MODEL_ID": m["id"],
        "CEVAL_MODEL_THINKING_LEVEL": m["thinking_level"],
        # Same model for the lookup subagent in every arm (PROTOCOL.md §8).
        "MECH_COMPACT_LOOKUP_MODEL": f"{m['provider']}/{m['id']}",
    })
    # The session-backed mech-compact (2026-10-06) writes no dumps and has no
    # designed-summary arm; it rejects their settings at load.
    if exp is not None and session_id is not None:
        # git must not discover the repo the runs/ tree lives in: the agent's
        # tools would otherwise see a real project's history (AGENTS.md: Safety).
        env["GIT_CEILING_DIRECTORIES"] = str(workdir(cfg, exp, arm, session_id))
        # Exactly the arm's extensions, as absolute paths (the bridge loads
        # these and nothing else), a private HOME, and pi-lcm's private store.
        env["CEVAL_EXTENSIONS"] = json.dumps([str(p) for p in extension_paths(
            cfg, arm, agent_cwd(cfg, exp, arm, session_id))])
        env["HOME"] = str(home_dir(cfg, exp, arm, session_id))
        if "pi-lcm" in arm_extensions(cfg, arm):
            env["LCM_DB_DIR"] = str(lcm_dir(cfg, exp, arm, session_id))
        if "pi-vcc" in arm_extensions(cfg, arm):
            env["PI_VCC_CONFIG_PATH"] = str(vcc_config_path(home_dir(cfg, exp, arm, session_id)))
    if "pi-vcc" in arm_extensions(cfg, arm):
        # pi-vcc defers every compaction but /pi-vcc to another package when this
        # is "fabric" (src/hooks/before-compact.ts); the study never sets it.
        env.pop("PI_FABRIC_COMPACTION_ENGINE", None)
    if "pi-fold" in arm_extensions(cfg, arm):
        # pi-fold takes options only in code (harness/extensions/pi-fold-study.ts).
        env["CEVAL_PI_FOLD_OPTIONS"] = json.dumps(cfg["fold"]["options"], sort_keys=True)
    env.update(cfg.arm_env(arm))
    env.update({k: str(v) for k, v in (overrides or {}).items()})
    return env


ARM_ENV_PREFIXES = ("MECH_COMPACT_", "LCM_", "CEVAL_PI_FOLD_", "PI_VCC_")
ARM_ENV_PATHS = ("MECH_COMPACT_DUMP_DIR", "LCM_DB_DIR", "PI_VCC_CONFIG_PATH")  # run-local paths, not arm settings


def arm_env_record(env: dict[str, str]) -> dict[str, str]:
    """The arm-defining part of an environment, recorded with every trial:
    extension settings (MECH_COMPACT_*, LCM_*) and the extensions loaded (by
    file or package name)."""
    out = {k: v for k, v in sorted(env.items()) if k.startswith(ARM_ENV_PREFIXES) and k not in ARM_ENV_PATHS}
    if "CEVAL_EXTENSIONS" in env:
        out["CEVAL_EXTENSIONS"] = ",".join(Path(p).name for p in json.loads(env["CEVAL_EXTENSIONS"]))
    return out


def check_pi_settings(cfg: Config) -> list[str]:
    """Warnings if pi's effective settings in the arms' cwd differ from the
    config (all arms must share them)."""
    path = Path(str(cfg["pi"]["settings_file"])).expanduser()
    warnings = []
    glob = {}
    if path.exists():
        glob = json.loads(path.read_text(encoding="utf-8"))
    else:
        warnings.append(f"pi settings file not found: {path}")
    # Resolve as pi's SettingsManager does: project compaction merged over global,
    # then compaction.modelOverrides["<provider>/<id>"] over both.
    comp = {**glob.get("compaction", {}), **project_settings(cfg)["compaction"]}
    model_key = f"{cfg['model']['provider']}/{cfg['model']['id']}"
    override = comp.get("modelOverrides", {}).get(model_key, {})
    if comp.get("enabled") is False:
        warnings.append("settings.compaction.enabled is false")
    for key, want in (("keepRecentTokens", cfg["pi"]["keep_recent_tokens"]),
                      ("reserveTokens", cfg["pi"]["reserve_tokens"])):
        if key in override and int(override[key]) != int(want):
            warnings.append(f"global compaction.modelOverrides[{model_key!r}].{key} = {override[key]} "
                            f"overrides the project setting; config says {want}")
    # Sampling: pi sends the model's samplingParams per request, so the study's
    # sampling setup is the study models file the bridge loads (pi.models_file),
    # not the server defaults. Verify it matches the config (PROTOCOL.md §8).
    want = cfg["model"].get("sampling_params")
    if want is not None:
        models_path = cfg.path(cfg["pi"]["models_file"])
        have = None
        if models_path.exists():
            try:
                models = json.loads(models_path.read_text(encoding="utf-8"))
                provider = models["providers"][cfg["model"]["provider"]]
                # The model is declared in `models` (a single-model llama-server
                # has no router catalog for pi to discover) or tuned through
                # `modelOverrides` (a model the provider discovers itself).
                entry = next((m for m in provider.get("models", []) if m.get("id") == cfg["model"]["id"]),
                             None) or provider["modelOverrides"][cfg["model"]["id"]]
                have = entry.get("samplingParams")
            except (json.JSONDecodeError, KeyError):
                have = "<unreadable models.json>"
        if have != want:
            warnings.append(f"models.json samplingParams for {cfg['model']['id']} = {have!r}, "
                            f"config says {want!r}")
    return warnings


# -----------------------------------------------------------------------------
# Bridge
# -----------------------------------------------------------------------------


@dataclass
class BridgeResult:
    ok: bool
    duration_ms: int
    error: str | None = None
    raw: dict = field(default_factory=dict)
    usage: dict | None = None      # model calls made during this bridge call (usageproxy), if active


# The runner's usage proxy (usageproxy.UsageProxy), if one is active: every
# bridge call then loads a copy of the study models file whose baseUrl points
# at the proxy, and the model calls it made (any caller inside pi) are summed.
_PROXY = None
_PROXY_MODELS: Path | None = None
_PROXY_AUTH: Path | None = None


def set_usage_proxy(cfg: Config, proxy, models_copy: Path | None = None) -> None:
    """Route pi's model calls through `proxy` (None: directly to the server)."""
    global _PROXY, _PROXY_MODELS, _PROXY_AUTH
    _PROXY, _PROXY_MODELS, _PROXY_AUTH = proxy, None, None
    if proxy is None:
        return
    src = cfg.path(cfg["pi"]["models_file"])
    data = json.loads(src.read_text(encoding="utf-8"))
    for prov in (data.get("providers") or {}).values():
        u = urlsplit(prov.get("baseUrl", ""))
        prov["baseUrl"] = f"http://127.0.0.1:{proxy.port}{u.path}"
    dst = (models_copy or (cfg.path(cfg["run"]["runs_dir"]) / ".usage-proxy-models.json")).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _PROXY_MODELS = dst
    # pi's llama.cpp provider takes the server address from its auth entry
    # (LLAMA_BASE_URL), not from the models file. So the bridge gets its own
    # auth file holding only that entry, pointed at the proxy: no credential is
    # copied (the entry has no key) and the user's auth.json is untouched.
    auth = {"llama.cpp": {"type": "api_key", "env": {"LLAMA_BASE_URL": f"http://127.0.0.1:{proxy.port}"}}}
    _PROXY_AUTH = dst.with_name(dst.stem + "-auth.json")
    _PROXY_AUTH.write_text(json.dumps(auth, indent=2), encoding="utf-8")


def call_bridge(cfg: Config, command: str, payload: dict, env: dict[str, str], cwd: Path,
                timeout_s: int) -> BridgeResult:
    if command in ("turn", "compact"):  # model calls; `context` makes none
        schedule.require_llm(cfg)  # backstop; runs stop between lines before this triggers
    if _PROXY is not None and command != "context":
        env = dict(env, CEVAL_PI_MODELS_FILE=str(_PROXY_MODELS), CEVAL_PI_AUTH_FILE=str(_PROXY_AUTH))
        mark = _PROXY.mark()
        br = _call_bridge(cfg, command, payload, env, cwd, timeout_s)
        br.usage = usageproxy.summarize(_PROXY.since(mark))
        return br
    return _call_bridge(cfg, command, payload, env, cwd, timeout_s)


def _call_bridge(cfg: Config, command: str, payload: dict, env: dict[str, str], cwd: Path,
                 timeout_s: int) -> BridgeResult:
    script = cfg.path(cfg["bridge"]["script"])
    t0 = time.monotonic()
    try:
        proc = subprocess.run([str(cfg["bridge"]["node_bin"]), str(script), command],
                              input=json.dumps(payload), capture_output=True, text=True,
                              env=env, cwd=cwd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return BridgeResult(False, int((time.monotonic() - t0) * 1000), f"timeout after {timeout_s}s")
    ms = int((time.monotonic() - t0) * 1000)
    last = (proc.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        raw = json.loads(last)
    except json.JSONDecodeError:
        raw = {}
    if proc.returncode != 0 or not raw.get("ok"):
        err = raw.get("error") or proc.stderr.strip()[-2000:] or f"exit {proc.returncode}"
        return BridgeResult(False, ms, err, raw)
    return BridgeResult(True, ms, None, raw)


# -----------------------------------------------------------------------------
# Turns and compactions, measured from the session file
# -----------------------------------------------------------------------------


@dataclass
class TurnResult:
    ok: bool
    error: str | None
    text: str
    calls: list[dict]
    tokens_in: int
    tokens_out: int
    duration_ms: int
    usage: dict | None = None


@dataclass
class CompactResult:
    ok: bool
    error: str | None
    summary_chars: int | None
    tokens_before: int | None
    first_kept_id: str | None
    details_kind: str | None
    duration_ms: int
    compactor: str | None = None   # who wrote the entry: pi / mech / lcm / vcc (pisession.compactor_of)
    details: dict | None = None    # small identifying part of the entry's details
    usage: dict | None = None      # model calls made by this operation (usageproxy), if measured


def prepare_fork(cfg: Config, main: Path, dst: Path, prune: bool = True) -> Path:
    """The session file a probe or continuation runs on (PROTOCOL.md v2 §6.1).
    Pruned (default): holding nothing the latest compaction summarised, for
    turns where the agent has file tools. Full (`prune=False`): a plain copy,
    for Study 1 probes, which have no built-in tools; context_lookup then reads
    the session's own history, as in real use. A full copy keeps the session
    id (pisession.copy_session: pi-lcm keys its store by it); a pruned fork
    gets a fresh one. Every runner calls this, so the fork policy lives in one
    place. The context pi builds from the fork must
    equal the main line's; a mismatch is a harness bug, so it raises rather
    than producing a trial."""
    fork = S.fork_pruned(main, dst) if prune else S.copy_session(main, dst)
    side = main.with_name(main.name + ".acp.json")  # billion-context-pi's compression state
    if side.exists():
        shutil.copyfile(side, fork.with_name(fork.name + ".acp.json"))
    a, b = context_hash(cfg, main), context_hash(cfg, fork)
    if a is not None and a != b:
        raise RuntimeError(f"pruned fork {fork} changes pi's context ({a} != {b})")
    return fork


def real_context_tokens(cfg: Config, session: Path) -> int | None:
    """Real tokens (the model's tokenizer, via the server's /tokenize: no
    generation) of the context messages pi builds from a session file. pi's own
    counts are chars/4 estimates that undercount dense material (PROTOCOL.md v2
    §5.1). Excludes the system prompt and tool definitions. None if unavailable."""
    import urllib.request
    br = call_bridge(cfg, "context", {"sessionFile": str(session.resolve()), "timeoutMs": 60_000, "includeText": True},
                     dict(os.environ, PI_PACKAGE_DIR=str(cfg["pi"]["package_dir"])), session.parent, 90)
    if not br.ok or "text" not in br.raw:
        return None
    base = cfg.raw.get("model", {}).get("tokenize_url", "http://127.0.0.1:8080/tokenize")
    try:
        text = "\n".join(_context_strings(json.loads(br.raw["text"])))
        req = urllib.request.Request(base, data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        return len(json.load(urllib.request.urlopen(req, timeout=120))["tokens"])
    except (OSError, ValueError, KeyError):
        return None


def _context_strings(x) -> list[str]:
    """The text a context message list carries (text, thinking, summaries, tool
    arguments and results), without JSON punctuation."""
    out: list[str] = []
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, list):
        for v in x:
            out += _context_strings(v)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k in ("text", "thinking", "summary", "content", "arguments"):
                out += [json.dumps(v)] if k == "arguments" and not isinstance(v, str) else _context_strings(v)
    return out


def context_hash(cfg: Config, session: Path) -> str | None:
    """sha256 of the context messages pi builds from a session file (no model
    call). None if the bridge cannot report it (e.g. pi is not installed)."""
    br = call_bridge(cfg, "context", {"sessionFile": str(session.resolve()), "timeoutMs": 60_000}, dict(
        os.environ, PI_PACKAGE_DIR=str(cfg["pi"]["package_dir"])), session.parent, 90)
    return br.raw.get("sha256") if br.ok else None


def turn(cfg: Config, session: Path, cwd: Path, env: dict[str, str], message: str,
         builtin_tools: bool = True, hidden_tools: tuple[str, ...] = (),
         follow_ups: tuple[str, ...] = ()) -> TurnResult:
    """builtin_tools=False removes pi's file/shell tools (extension tools such as
    context_lookup stay): probes must not read session files from disk.
    hidden_tools removes named extension tools for this turn (probes hide
    dump_context, PROTOCOL.md v2 §6.1)."""
    n0 = S.line_count(session)
    payload = {"sessionFile": str(session), "cwd": str(cwd), "message": message,
               "timeoutMs": cfg["bridge"]["turn_timeout_s"] * 1000}
    if not builtin_tools:
        payload["noBuiltinTools"] = True
    if hidden_tools:
        payload["hiddenTools"] = list(hidden_tools)
    if follow_ups:
        payload["followUps"] = list(follow_ups)
    br = call_bridge(cfg, "turn", payload,
                     env, cwd, cfg["bridge"]["turn_timeout_s"] + 60)
    delta = S.entries_after(session, n0)
    tin, tout = S.token_usage(delta)
    return TurnResult(br.ok, br.error, S.final_assistant_text(delta), S.tool_calls(delta), tin, tout, br.duration_ms,
                      br.usage)


def continue_turn(cfg: Config, session: Path, cwd: Path, env: dict[str, str], builtin_tools: bool = True) -> TurnResult:
    """Resume a session whose last entry is a tool result, with no new message
    (pi's agent.continue()). Used to branch a probe turn after a lookup."""
    n0 = S.line_count(session)
    payload = {"sessionFile": str(session), "cwd": str(cwd), "timeoutMs": cfg["bridge"]["turn_timeout_s"] * 1000}
    if not builtin_tools:
        payload["noBuiltinTools"] = True
    br = call_bridge(cfg, "continue", payload, env, cwd, cfg["bridge"]["turn_timeout_s"] + 60)
    delta = S.entries_after(session, n0)
    tin, tout = S.token_usage(delta)
    return TurnResult(br.ok, br.error, S.final_assistant_text(delta), S.tool_calls(delta), tin, tout, br.duration_ms,
                      br.usage)


_WORD = re.compile(r"[^\s'\"`;|&<>(){}]+")


def _path_tokens(s: str) -> list[str]:
    """Whole shell words that look like paths: containing `/`, or starting with `~`
    or `..` (an option's value after `=` counts). A relative word such as `deploy/`
    resolves under the working directory; it is not a search of `/`."""
    out = []
    for w in _WORD.findall(s):
        w = w.split("=", 1)[1] if w.startswith("-") and "=" in w else w
        if "/" in w or w.startswith("~") or w.startswith(".."):
            out.append(w)
    return out
_BROAD = {"/", "/work", "/root", "/home", "~", "/tmp"}


# pi's built-in file and shell tools: the only tools that reach the disk at
# the agent's choice. Extension tools (context_lookup, lcm_grep/lcm_describe/
# lcm_expand, pi_fold_context, vcc_recall) are the arms' own recovery mechanisms; their
# arguments (e.g. lcm_grep's regex `pattern`) are not paths, and their use is
# recorded separately (recovery_tool_calls).
BUILTIN_FILE_TOOLS = frozenset({"read", "bash", "edit", "write", "grep", "find", "ls"})


def audit_view(calls: list[dict]) -> list[dict]:
    """The parts of built-in tool calls that address files, for the leak audit:
    the path of file tools and the shell command, with a free-standing `/` (division in
    code, as in `x / 1000`) masked. File contents written by edit/write calls
    are not paths: their `/` operators and URLs would otherwise read as
    searches of the root directory."""
    out = []
    for c in calls:
        if c.get("name") not in BUILTIN_FILE_TOOLS:
            continue
        args = c.get("args") or {}
        if c.get("name") == "bash":
            # Mask a free-standing `/` (or //, /=) only inside quoted code such as
            # python -c '... 10 / 2 ...'; an unquoted `find /` is still a search.
            # A heredoc body is stdin content (code), like an edit's contents: drop it.
            cmd = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\b", "<<HEREDOC", str(args.get("command", "")), flags=re.S)
            view = {"command": re.sub(r"'[^']*'|\"[^\"]*\"",
                                      lambda m: re.sub(r"(?<![\w.~-])//?=?(?![\w.~-])", " ", m.group(0)), cmd)}
        else:
            view = {k: v for k, v in args.items() if k in ("path", "file", "directory", "dir", "pattern", "glob")}
        out.append({"name": c.get("name"), "args": view})
    return out


def leak_paths(calls: list[dict], wd: Path, allowed: tuple[Path, ...] = (), root: Path | None = None) -> list[str]:
    """Paths in tool-call arguments that could reach removed context (PROTOCOL.md
    v2 §6.1): anything in the repo tree (runs/, session files, dumps) or pi's
    agent dir that is not inside the agent's working directory or an allowed
    directory (the dumps, in lookup arms), and searches rooted at broad
    directories such as `/` or `/work`. System paths (/usr, /etc, /proc, ...)
    are fine. Conservative by design: a flagged trial is excluded and reported."""
    root = (root or repo_root()).resolve()
    wd = wd.resolve()
    ok_dirs = [wd, *(Path(a).resolve() for a in allowed)]
    pi_dir = (Path.home() / ".pi").resolve()
    hits = []

    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)

    for c in audit_view(calls):
        for s in strings(c.get("args") or {}):
            for tok in _path_tokens(s):
                if tok.rstrip("/") in _BROAD or tok in _BROAD:
                    hits.append(tok)
                    continue
                p = Path(tok).expanduser()
                p = (p if p.is_absolute() else wd / p).resolve()
                if any(p == d or d in p.parents for d in ok_dirs):
                    continue
                if p == root or root in p.parents or p == pi_dir or pi_dir in p.parents:
                    hits.append(tok)
    return hits


def audit_leak(calls: list[dict], wd: Path, allowed: tuple[Path, ...] = ()) -> bool:
    """True if any tool call could reach removed context (see leak_paths)."""
    return bool(leak_paths(calls, wd, allowed))


def compact(cfg: Config, session: Path, cwd: Path, env: dict[str, str], timeout_s: int | None = None) -> CompactResult:
    """One pi compaction (session.compact()). timeout_s: the arm's budget
    (compact_timeout_s; pi-lcm's hierarchical summarisation takes many calls)."""
    timeout_s = int(timeout_s or cfg["bridge"]["compact_timeout_s"])
    n0 = S.line_count(session)
    br = call_bridge(cfg, "compact", {"sessionFile": str(session), "cwd": str(cwd),
                                      "timeoutMs": timeout_s * 1000},
                     env, cwd, timeout_s + 60)
    comp = S.last_compaction(S.entries_after(session, n0))
    if comp is None:
        return CompactResult(False, br.error or "no compaction entry written", None, None, None, None, br.duration_ms,
                             usage=br.usage)
    details = comp.get("details") or {}
    small = {k: details[k] for k in ("kind", "lcm", "stats", "arm") if k in details}
    small.update(S.vcc_details(details))
    return CompactResult(br.ok, br.error, len(comp.get("summary", "") or ""), comp.get("tokensBefore"),
                         comp.get("firstKeptEntryId"), details.get("kind"), br.duration_ms,
                         S.compactor_of(comp), small, br.usage)


def fold_commit(cfg: Config, session: Path, cwd: Path, env: dict[str, str]) -> CompactResult:
    """pi-fold's per-round commit turn on the main line (docs/build/comparison-arms.md).

    pi-fold never uses pi's compaction: it folds in its `context` hook, and
    only after a provider-measured reply whose measurement receipt it wrote at
    message_end in this process. The main line is synthetic (no live replies),
    so the commit turn is fold.commit_messages run in ONE bridge process with
    no built-in tools: the first reply is the measurement, the second prompt's
    context hook commits and sends the folded request. Measured from the
    session file: the fold records written (compactor 'fold' if any, else
    'none'), the provider prompt size of the first reply (before) and of the
    last (after, the folded context)."""
    if cfg["fold"].get("commit") == "command":
        return fold_command(cfg, session, cwd, env)
    msgs = list(cfg["fold"]["commit_messages"])
    n0 = S.line_count(session)
    tr = turn(cfg, session, cwd, env, msgs[0], builtin_tools=False, follow_ups=tuple(msgs[1:]))
    delta = S.entries_after(session, n0)
    summ = S.fold_summary(delta)
    replies = [m for m in S.messages(delta) if m.get("role") == "assistant"]
    summ["provider_tokens_before"] = S.context_tokens(replies[0]) if replies else None
    summ["provider_tokens_after"] = S.context_tokens(replies[-1]) if len(replies) > 1 else None
    summ["replies"] = [S.final_assistant_text([{"type": "message", "message": m}]) for m in replies]
    return CompactResult(tr.ok, tr.error, summ["placeholder_chars"] if summ["folds"] else None,
                         summ["provider_tokens_before"], None, "pi-fold", tr.duration_ms,
                         "fold" if summ["folds"] else "none", summ, tr.usage)


def billion_commit(cfg: Config, session: Path, cwd: Path, env: dict[str, str]) -> CompactResult:
    """billion-context-pi's commit turn on the main line, at the point where other
    arms compact. It cancels pi's compaction and compresses only when the model
    calls its `compress` tool in a live turn, nudged by the extension once the
    context passes its thresholds. The synthetic main line has no live turns, so
    each round ends with one neutral turn (the same checkpoint message as
    pi-fold's). Measured from the session file and its sidecar
    (<session>.acp.json): compress calls, active blocks and compressed tokens;
    compactor 'billion' if an active block exists, else 'none'."""
    n0 = S.line_count(session)
    tr = turn(cfg, session, cwd, env, cfg["fold"]["commit_messages"][0], builtin_tools=False)
    delta = S.entries_after(session, n0)
    replies = [m for m in S.messages(delta) if m.get("role") == "assistant"]
    side = session.with_name(session.name + ".acp.json")
    blocks = []
    if side.exists():
        try:
            blocks = [b for b in json.loads(side.read_text(encoding="utf-8")).get("blocks", []) if b.get("active")]
        except (OSError, ValueError):
            blocks = []
    calls = [c for c in S.tool_calls(delta) if c.get("name") == "compress"]
    summ = {"compress_calls": len(calls), "active_blocks": len(blocks),
            "compressed_tokens": sum(int(b.get("compressedTokens") or 0) for b in blocks),
            "summary_chars": sum(len(b.get("summary") or "") for b in blocks),
            "provider_tokens_before": S.context_tokens(replies[0]) if replies else None,
            "provider_tokens_after": S.context_tokens(replies[-1]) if len(replies) > 1 else None,
            "replies": len(replies)}
    return CompactResult(tr.ok, tr.error, summ["summary_chars"] if blocks else None, summ["provider_tokens_before"],
                         None, "billion-context-pi", tr.duration_ms, "billion" if blocks else "none", summ, tr.usage)


def fold_command(cfg: Config, session: Path, cwd: Path, env: dict[str, str]) -> CompactResult:
    """pi-fold's forced fold (decided 2026-10-06): its own `/fold` command on the
    main line, at the point where the other arms compact (their /compact). Bare
    /fold commits every staged mark now, at any occupancy, and makes no model
    call (the bridge's `command` op refuses one). Without a measured reply there
    is no automatic top-up beyond the staged marks. The folds take effect in the
    context of the next model call (the probe); their size is in the fold
    records, and the probe's provider prompt size shows the folded context."""
    n0 = S.line_count(session)
    br = call_bridge(cfg, "command", {"sessionFile": str(session), "cwd": str(cwd),
                                      "command": cfg["fold"].get("command", "/fold"),
                                      "timeoutMs": cfg["bridge"]["turn_timeout_s"] * 1000},
                     env, cwd, cfg["bridge"]["turn_timeout_s"] + 60)
    ms = br.duration_ms
    summ = S.fold_summary(S.entries_after(session, n0))
    summ["provider_tokens_before"] = summ["provider_tokens_after"] = None
    return CompactResult(br.ok, br.error, summ["placeholder_chars"] if summ["folds"] else None,
                         None, None, "pi-fold", ms, "fold" if summ["folds"] else "none", summ, br.usage)


def cut_ok(session: Path, first_kept_id: str | None, early_ids: list[str], late_ids: list[str]) -> bool | None:
    """True if every early fact entry lies before the cut and every late one at/after it."""
    if not first_kept_id:
        return None
    idx = S.entry_index(S.read_entries(session))
    if first_kept_id not in idx:
        return None
    cut = idx[first_kept_id]
    early_ok = all(idx.get(e, -1) < cut for e in early_ids if e in idx)
    late_ok = all(idx.get(e, 10**9) >= cut for e in late_ids if e in idx)
    return early_ok and late_ok
