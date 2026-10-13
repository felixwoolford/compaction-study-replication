# Verbatim compaction: replication package (Studies 1 and 4)

This package re-runs the two main studies of the paper on compaction for the [pi coding agent](https://github.com/earendil-works/pi-coding-agent) (arXiv link to be added) with your own model and pi setup. It compares three ways of compacting an agent's context:

- **verbatim** ([felixwoolford/verbatim-compact-pi](https://github.com/felixwoolford/verbatim-compact-pi)) keeps the user's and agent's words verbatim and makes the full history retrievable.
- **pi-default** is pi's built-in compaction: the model writes a summary.
- **pi-vcc** ([monotykamary/pi-vcc](https://github.com/monotykamary/pi-vcc)) is a third-party compactor that builds its summary by rule-based extraction.

The two studies:

| study | material | what is measured | scoring |
|---|---|---|---|
| **Study 1: planted facts** | 18 generated coding sessions, each compacted 3 times | after each compaction, can the agent recall details planted earlier, and does it answer wrongly with confidence? | by rule; no external data or API needed |
| **Study 4: commitments** | 27 probes on 18 real pi sessions from a public dataset | after compaction, does the agent respect an earlier commitment (a user constraint or an agreed decision) when a new instruction tests it? | a blind LLM judge (any OpenAI-compatible endpoint) |

The defaults reproduce the paper's main runs: Qwen 3.8 27B (`unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_XL`) on a local llama.cpp server, with thinking at `xhigh`.

---

## Quick start

Run all commands from the repository root. Each script changes to the right directory itself.

```bash
uv sync                                     # Python environment (uv installs Python 3.14 if needed)
$EDITOR replication.config.json             # your pi install, your model, your judge (see "Configuration")
uv run python scripts/setup.py              # generates config/; clones verbatim and pi-vcc into vendor/repos/
```

**Check the installation first, with no model and no API key (about 10 minutes):**

```bash
uv run python scripts/download_sessions.py  # fetches Study 4's 18 sessions; checks SHA-256
uv run python scripts/run_s4.py replay      # compacts the sessions (pi-default calls your model; see note)
uv run python scripts/verify_replays.py     # expected: "identical to the paper" for verbatim and pi-vcc
```

The verbatim and pi-vcc replays make no model calls. pi-default's replay calls your model to write its summaries; to skip that part of the check, temporarily set `"study4": {"arms": ["verbatim", "pi-vcc"]}`. If `verify_replays.py` reports both as identical, pi, the two extensions and the session data are all working.

**Study 1:**

```bash
uv run python scripts/run_s1.py             # runs the configured arms on the 18 sessions (resumable)
uv run python scripts/analyze_s1.py         # scores by rule; writes results/<experiment>/replication-analysis.json
```

**Study 4:**

```bash
uv run python scripts/run_s4.py replay      # if not already done
uv run python scripts/run_s4.py run         # probe turns: arms × 27 probes × 3 repetitions (resumable)
JUDGE_API_KEY=... uv run python scripts/score_s4.py    # blind judging; arm labels are hidden from the judge
uv run python scripts/analyze_s4.py         # writes results/S4/replication-analysis.json
```

Every step can be interrupted and resumed by running the same command again. Finished work is skipped. To start over, delete `runs/` and `results/`.

---

## Requirements

- **pi** (`@earendil-works/pi-coding-agent`), installed however you like.
  - The paper used 1.0.3 for Study 1 and 1.1.0 for Study 4; `setup.py` warns if your version differs.
  - The package drives pi in-process through its SDK (`harness/bridge/pi-bridge.mjs`).
  - Each run session gets its own working directory under `runs/`, and only the arm's own extension is loaded. Your pi settings in `agent_dir` are only read (for authentication), never changed.
- **Node**, to run pi.
- **git**, to clone the two extensions.
- **[uv](https://docs.astral.sh/uv/)** for Python.
- **Network access** to GitHub (the extensions) and Hugging Face (the sessions).
- **A model pi can reach**, either:
  - a local **OpenAI-compatible server** (llama.cpp, vLLM, …) with a context of at least 128k tokens and one request at a time; or
  - a **provider pi already supports**. Set `"kind": "builtin"`. Sampling cannot be controlled then.
- **For Study 4:** an API key for any OpenAI-compatible chat-completions endpoint, used as the judge.

**Run time on one GPU server** with Qwen at xhigh. A model that thinks less is faster; Gemma 4 31B took about a third of the time.

| study | time |
|---|---|
| Study 1 | about 4 h per arm |
| Study 4 replay | about 1–2 h (pi-default's summaries; the other arms take minutes) |
| Study 4 probe turns | about 1.5–2 h per arm for 3 repetitions |

---

## Configuration

`replication.config.json` is the only file you need to edit. `scripts/setup.py` turns it into the harness config `config/eval.config.json` and the pi models files `config/pi-models*.json`. Re-run the setup after every edit.

| section | what it sets | paper's value |
|---|---|---|
| `extensions` | repository and commit for verbatim and pi-vcc | verbatim `6b970d3`, pi-vcc `fd2603e` (release 0.8.12) |
| `pi` | `package_dir` (the installed pi package), `agent_dir` (your pi auth and settings, read only), `version` | 1.1.0 |
| `model.kind` | `openai-completions` for a server you run, `builtin` for a provider pi knows | `openai-completions` |
| `model.provider`, `id`, `base_url`, `api_key` | how pi reaches the model | `llama.cpp`, Qwen 3.8 27B, `http://127.0.0.1:8080/v1` |
| `model.thinking_level`, `thinking_level_map`, `chat_template_kwargs` | reasoning effort and how pi passes it to the chat template | `xhigh` |
| `model.s1_sampling` | Study 1 sampling | temperature 1.0, top_k 20, top_p 0.95, seed 20261004 |
| `model.s4_temperature` | Study 4 temperature (each repetition uses its own seed) | 0.7 |
| `study1` | `arms`, `sessions` range (0–17), `experiment` name (the results folder) | all three arms |
| `study4` | `arms`, `seeds` (one per repetition), `skip_probes` | seeds 20261101–03; the 2 probes that failed their controls in the paper are skipped |
| `judge` | `base_url`, `model`, `api_key_env`, `temperature` | Claude Sonnet (see below) |

**Example: a provider pi already knows.** pi supplies the endpoint and authentication from your `agent_dir`.

```json
"model": {"kind": "builtin", "provider": "openai-codex", "id": "gpt-6-luna", "thinking_level": "medium"}
```

**Example: a different local model.** Use the thinking levels its chat template supports. Gemma 4, for example, only switches thinking on or off.

```json
"model": {"kind": "openai-completions", "provider": "llama.cpp", "id": "unsloth/gemma-4-31B-it-GGUF:UD-Q5_K_XL",
          "base_url": "http://127.0.0.1:8080/v1", "api_key": "local", "reasoning": true, "thinking_level": "medium",
          "thinking_level_map": {"off": "off", "minimal": null, "low": null, "medium": "medium", "high": null, "xhigh": null, "max": null},
          "chat_template_kwargs": {"enable_thinking": {"$var": "thinking.enabled"}, "preserve_thinking": true},
          "input": ["text", "image"], "s1_sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 20261004},
          "s4_temperature": 1.0}
```

**The judge.**
- The default is Claude Sonnet through Anthropic's OpenAI-compatible endpoint, the model family that judged the paper's replies. Any OpenAI-compatible endpoint works.
- The judge sees the frozen brief (`materials/s4/scorer.md`) and one reply at a time, with no arm label. It returns `pass`, plus `confident` for failures.
- Scores are cached, so an interrupted run resumes.

**The models files** written by the setup:

| file | declared context | used for |
|---|---|---|
| `config/pi-models.json` | 128k | Study 1, and all Study 4 probe turns |
| `config/pi-models-48k.json` | 48k (reserve 8,192, keep 8,000) | Study 4's model-free replays |
| `config/pi-models-48k-r32k.json` | 72,576 (reserve 32,768, keep 8,000) | Study 4's pi-default replay; see "Differences from the paper" |

---

## Outputs

| study | file | contents |
|---|---|---|
| 1 | `results/<experiment>/trials.jsonl`, `compactions.jsonl` | every probe answer and every compaction |
| 1 | `results/<experiment>/scored.jsonl` | answers scored by rule |
| 1 | `results/<experiment>/replication-analysis.json` | recall and confident-wrong rates per arm; paired sign-flip tests across sessions of verbatim against each other arm |
| 4 | `runs/S4/replay/rows-<arm>-48k.jsonl` | every replay compaction |
| 4 | `results/S4/runs-rep<i>.jsonl` | every probe turn, one file per repetition |
| 4 | `runs/S4/replies-rep<i>/` | the replies |
| 4 | `runs/S4/blind-rep<i>/` | the judge's scores and the key that maps them back to arms |
| 4 | `results/S4/replication-analysis.json` | pass and confident-wrong rates per arm; paired sign-flip tests across probes for every pair of arms |

Study 4 runs its probes through the harness's `study3` module: it was built on Study 3's probe pipeline, which Study 4 reused unchanged.

---

## What to expect

**Reference values from the paper.** Study 4 figures are on 27 probes, repetitions 1–3, with failed turns counted as failures.

| Study 1 recall | verbatim | pi-default | pi-vcc |
|---|---|---|---|
| Qwen 3.8 27B | 97.8%* | 75.3% | 98.2% |
| GPT-6 Luna | 95.1% | 55.0% | 45.6% |
| Gemma 4 31B | 93.3% | 51.5% | 67.5% |

\* The paper's Study 1 on Qwen ran an earlier version of verbatim (with pi 1.0.3), before the release at `6b970d3`. This package runs the release for every study and model. Its Study 1 performance is expected to be equivalent: on Luna and Gemma the release recalled 95.1% and 93.3%.

| Study 4 pass rate | verbatim | pi-default | pi-vcc |
|---|---|---|---|
| Qwen 3.8 27B | 74.1% | 64.2% | 53.1% |
| GPT-6 Luna | 46.9% | 30.9% | 28.4% |
| GPT-6.1 Sol | 69.1% | 74.1% | 58.0% |
| Gemma 4 31B | 39.5% | 35.8% | 32.1% |

**Expect noise.**
- Sampling is seeded, but llama.cpp output is not bit-reproducible across hardware or builds.
- In the paper, about 14% of repeated answers flipped from one repetition to the next.
- Compare per-arm means and the direction of the differences, not individual probes.
- The model-free replays (`verify_replays.py`) should match exactly.

## Differences from the paper

- **Study 1 on Qwen** used an earlier version of verbatim; see the note under the Study 1 table.
- **pi-default's Study 4 replay** declares 72,576 tokens with a 32,768-token reserve.
  - The trigger point (39,808 tokens) and the 8,000 kept tokens are the same as for the other arms.
  - At the stock 48k settings, pi caps its summary at 0.8 × reserve = 6,553 tokens. A reasoning model at high effort can spend all of that on thinking, and the compaction then fails: 15 of 16 sessions for Qwen at xhigh.
  - The paper's Luna and Sol replays used the stock 48k settings, but their provider sends no output limit, so the cap did not apply.
- **The judge.**
  - The paper's replies were judged blind by Claude Sonnet agents with the same brief; agreement between two judges was κ ≈ 0.85–1.0.
  - A judge on another model will not agree perfectly.
- **Timeouts.**
  - A probe turn that runs past its limit (15 minutes in Study 4, 30 in Study 1) is recorded as failed. Some models loop on repeated tool calls; Gemma did on one probe.
  - Failed turns count as failures in the analysis.
- **Synthetic timestamps.**
  - Study 1's generated sessions carry message timestamps ahead of the wall clock. They were kept so the runs stay comparable.
  - This can interact with pi's output-token estimate for extensions that rewrite the context. It did for billion-context-pi in the paper; it does not affect the three methods here.

## Troubleshooting

- **`verify_replays.py` reports a difference.** Check the extension commits and your pi version (`setup.py` prints both). The replays depend on pi's compaction cut logic.
- **pi-default compactions fail with "generation hit the token cap".** Your model is thinking past pi's summary budget. Lower the thinking level for the run, or check that the 72,576 / 32,768 models file is in use (it is by default).
- **The server rejects requests for context size.** The server must hold at least 128k tokens of context per request.
- **The server rejects image input.** Some sessions contain images. Use a model or server with image support, or remove `"image"` from `model.input` and expect those turns to fail.
- **Judge errors.** Check `JUDGE_API_KEY` (or the variable named in `judge.api_key_env`) and `judge.base_url`. The script retries 5 times, then stops; rerun it to resume.

---

## Data and licences

- **The code** in this repository is under the MIT licence (`LICENSE`).
- **Study 4's sessions** come from the public Hugging Face dataset [MaxDevv/real-pi-coding-agent-traces-sessions](https://huggingface.co/datasets/MaxDevv/real-pi-coding-agent-traces-sessions).
  - Each session stays under its source's licence. All 18 used here are MIT-licensed.
  - They are not redistributed here. `scripts/download_sessions.py` fetches them and checks their hashes.
- **The probes** (`materials/s4/probes.json`) were written for this study and quote short strings from those sessions. See `NOTICE.md`.
- **Study 1's material** is fully synthetic, generated from `materials/scenarios/s1.json`.
- **verbatim and pi-vcc** are cloned from their own repositories and are under their own licences.

## Layout

| path | contents |
|---|---|
| `replication.config.json` | your settings |
| `scripts/` | `setup`, `download_sessions`, `run_s1`, `analyze_s1`, `run_s4`, `verify_replays`, `score_s4`, `analyze_s4` |
| `harness/src/ceval/` | the evaluation harness the paper used |
| `harness/bridge/pi-bridge.mjs` | drives pi through its SDK |
| `materials/` | the Study 1 scenario, the probe prompt, Study 4's probes and session selection, the judge brief, the expected replays |
| `config/eval.config.template.json` | the template that `setup.py` fills in |
| `vendor/repos/`, `config/*.json`, `runs/`, `results/` | created by the scripts (ignored by git) |
