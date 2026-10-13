"""Turn replication.config.json into the harness config (config/eval.config.json) and the pi models files
(config/pi-models*.json), clone verbatim and pi-vcc from their repositories at the configured commits into
vendor/repos/, and check pi and node.
  uv run python scripts/setup.py"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "replication.config.json").read_text())

# Study 4 replays: a 48k declaration (keep 8,000, reserve 8,192) as in the paper; pi-default's replay keeps the same
# trigger point (window - reserve = 39,808) and keep, but declares 72,576 with reserve 32,768 so pi's summary cap
# (0.8 x reserve) leaves room for a reasoning model's thinking (PROTOCOL 18.3b).
MODELS_FILES = {"pi-models.json": 128_000, "pi-models-48k.json": 48_000, "pi-models-48k-r32k.json": 72_576}


def model_entry(m: dict, window: int, sampling: dict | None) -> dict:
    e = {"id": m["id"], "name": m["id"], "contextWindow": window, "maxTokens": {128_000: 128_000}.get(window, 32_768),
         "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}, "reasoning": m.get("reasoning", True),
         "thinkingLevelMap": m.get("thinking_level_map"), "input": m.get("input", ["text"]),
         "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": True, "supportsStrictMode": False, "maxTokensField": "max_tokens",
                    "thinkingFormat": "chat-template", "chatTemplateKwargs": m.get("chat_template_kwargs", {})}}
    if sampling:
        e["samplingParams"] = sampling
    return e


def models_file(window: int) -> dict:
    m = CFG["model"]
    if m["kind"] == "builtin":
        return {"providers": {m["provider"]: {"modelOverrides": {m["id"]: {"contextWindow": window}}}}}
    return {"providers": {m["provider"]: {"baseUrl": m["base_url"], "api": "openai-completions", "apiKey": m.get("api_key", "local"),
                                          "models": [model_entry(m, window, m.get("s1_sampling"))]}}}


def eval_config() -> dict:
    tpl = json.loads((ROOT / "config/eval.config.template.json").read_text())
    pi, m = CFG["pi"], CFG["model"]
    agent_dir = Path(pi["agent_dir"]).expanduser()
    tpl["pi"].update(bin=shutil.which(pi["bin"]) or pi["bin"], package_dir=pi["package_dir"],
                     settings_file=str(agent_dir / "settings.json"), version=pi["version"])
    tpl["bridge"]["node_bin"] = shutil.which(CFG["node_bin"]) or CFG["node_bin"]
    tpl["model"] = {"provider": m["provider"], "id": m["id"], "context_window": 128_000, "thinking_level": m["thinking_level"]}
    if m["kind"] != "builtin" and m.get("s1_sampling"):
        tpl["model"]["sampling_params"] = m["s1_sampling"]
    tpl["usage_proxy"]["enabled"] = m["kind"] != "builtin"  # the proxy needs a baseUrl to point pi at
    if m["kind"] != "builtin":
        tpl["usage_proxy"]["upstream"] = m["base_url"].rstrip("/").removesuffix("/v1")
    vb = ROOT / "vendor/repos/verbatim"
    pkg = json.loads((vb / "package.json").read_text())["pi"]
    tpl["extension"]["registry"]["verbatim"]["source"] = str(vb / pkg["extensions"][0])
    tpl["arms"]["verbatim"]["skill"] = str(vb / pkg["skills"][0])
    tpl["extension"]["registry"]["pi-vcc"]["source"] = str(ROOT / "vendor/repos/pi-vcc")
    return tpl


def clone(name: str, spec: dict) -> str:
    """Clone (or update) a repository into vendor/repos/<name> and check out the configured commit."""
    d = ROOT / "vendor/repos" / name
    if not (d / ".git").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", spec["repo"], str(d)], check=True)
    else:
        subprocess.run(["git", "-C", str(d), "fetch", "--quiet", "--all"], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "--quiet", spec["commit"]], check=True)
    return subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    for name, spec in ((k, v) for k, v in CFG["extensions"].items() if not k.startswith("_")):
        print(f"{name}: {spec['repo']} at {clone(name, spec)}")
    # pi-vcc loads in place from its clone; its only dependencies are pi's own packages, which pi provides to
    # extensions. verbatim is a single file copied into each session's working directory.
    for name, window in MODELS_FILES.items():
        (ROOT / "config" / name).write_text(json.dumps(models_file(window), indent=2) + "\n")
    (ROOT / "config/eval.config.json").write_text(json.dumps(eval_config(), indent=2) + "\n")
    print("wrote config/eval.config.json and", ", ".join(f"config/{n}" for n in MODELS_FILES))
    problems = []
    if not Path(CFG["pi"]["package_dir"], "package.json").exists():
        problems.append(f"pi package not found at {CFG['pi']['package_dir']}")
    else:
        v = json.loads(Path(CFG["pi"]["package_dir"], "package.json").read_text())["version"]
        if v != CFG["pi"]["version"]:
            problems.append(f"pi version is {v}, config says {CFG['pi']['version']} (the paper used 1.0.3 for Study 1 and 1.1.0 for Study 4)")
    if not shutil.which(CFG["node_bin"]):
        problems.append(f"node not found: {CFG['node_bin']}")
    for p in problems:
        print("WARNING:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
