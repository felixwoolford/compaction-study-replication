"""Run E1/E2: build main session lines per arm, compact, fork, probe. Resumable.

Outputs (append-only JSONL):
  results/<exp>/trials.jsonl       one line per (session, arm, round, rep, mode, fact)
  results/<exp>/compactions.jsonl  one line per compaction
  runs/<exp>/plans/<sid>.json      the exact stimulus plan used
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import driver as D
from . import usageproxy
from . import pisession as S
from . import schedule
from .config import Config
from .scoring import parse_answers
from .challenge import build_ec_plan
from .stimuli import Plan, build_plan, scenario_indices


class _Stop(Exception):
    """The model may not be used now; stop before the next line."""


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


PROBE_FOLLOWUP = ("No tools are available for this check. Reply now with only the answer lines, "
                  "one per question in the form `A<n>: <answer>`, using UNKNOWN where you cannot answer.")

# v2 (PROTOCOL.md §5.3): one follow-up when some questions in a batch have no answer line.
PARTIAL_FOLLOWUP = ("Your reply has no answer line for {missing}. Reply now with one line for each of "
                    "these questions, in the form `A<n>: <answer>`, using UNKNOWN only as a last resort.")


class Runner:
    def __init__(self, cfg: Config, scenario: dict):
        self.cfg = cfg
        self.sc = scenario
        self.exp = scenario["experiment"]
        self.res_dir = D.results_dir(cfg, self.exp)
        self.runs_dir = cfg.path(cfg["run"]["runs_dir"]) / self.exp
        self.trials_path = self.res_dir / "trials.jsonl"
        self.comp_path = self.res_dir / "compactions.jsonl"
        self.probe_tpl = cfg.path("materials/prompts/probe.md").read_text(encoding="utf-8")
        self.meta = {"git_commit": _git_commit(cfg.root), "pi_version": cfg["pi"]["version"],
                     "model": f"{cfg['model']['provider']}/{cfg['model']['id']}"}
        self.done = {self._key(t) for t in _read_jsonl(self.trials_path)}
        # Lines stopped by a failed compaction are finished: a failure is data
        # and is never retried (PROTOCOL.md §9.4). Without this, a resume would
        # treat the stopped line as unfinished and redo it from scratch
        # (happened once, S1B 2026-10-07 14:25-15:58; see DEVIATIONS.md).
        self.stopped = {(c["session_id"], c["arm"]) for c in _read_jsonl(self.comp_path) if not c.get("ok")}
        # Continuation study (Study 1b): each line is copied from the source
        # experiment's line after `after_round` and continued (_run_continue).
        self.cont = scenario.get("continue_from")
        self.row_extra: dict = {}  # fields added to every row of the current line

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _key(t: dict) -> tuple:
        return (t["session_id"], t["arm"], t["round"], t["rep"], t["probe_mode"])

    def plans(self, indices: list[int] | None = None) -> list[Plan]:
        out = []
        allowed = scenario_indices(self.sc)
        if indices is not None and self.cont and set(indices) - set(allowed):
            raise ValueError(f"{self.exp} runs sessions {allowed} only, not {sorted(set(indices) - set(allowed))}")
        for i in indices if indices is not None else allowed:
            gen = self.sc.get("generator")
            if gen == "stories":
                from .stories import build_story_plan as builder
            else:
                builder = build_ec_plan if gen == "challenge" else build_plan
            p = builder(self.sc, i, keep_recent_tokens=self.cfg.keep_recent_tokens,
                        chars_per_token=self.cfg.chars_per_token)
            path = self.runs_dir / "plans" / f"{p.session_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(p.to_dict(), indent=1), encoding="utf-8")
            out.append(p)
        return out

    def _write_phases(self, plan: Plan, session: Path, cwd: Path, names: list[str]) -> dict[str, list[str]]:
        """Append the named phases; return {fact_uid: [entry ids]} for planted steps."""
        fact_ids: dict[str, list[str]] = {}
        for name in names:
            idx = next(i for i, ph in enumerate(plan.phases) if ph["name"] == name)
            w = S.SessionWriter(session, cwd=str(cwd), version=int(self.cfg["pi"]["session_version"]),
                                provider=self.cfg["model"]["provider"], model=self.cfg["model"]["id"],
                                seed=plan.seed * 100 + idx,
                                thinking_level=self.cfg["model"]["thinking_level"])
            for st in plan.phases[idx]["steps"]:
                ids = w.step(st)
                if st.get("fact_uid"):
                    fact_ids.setdefault(st["fact_uid"], []).extend(ids)
        return fact_ids

    def _est_tokens(self, plan: Plan, names: list[str]) -> int:
        from .filler import step_chars
        chars = sum(step_chars(st) for ph in plan.phases if ph["name"] in names for st in ph["steps"])
        return int(chars / self.cfg.chars_per_token)

    def _probe_sets(self, plan: Plan, mode: str, rnd: int = 1) -> list[list[str]]:
        """Question batches at round `rnd`: facts planted by then (round_planted),
        in probe order, `probe_batch_size` at a time (default: all at once)."""
        planted = {f.uid for f in plan.facts if f.round_planted <= rnd}
        order = [u for u in plan.probe_order if u in planted]
        if mode != "batched":
            return [[uid] for uid in order]
        size = self.sc.get("probe_batch_size") or len(order) or 1
        return [order[i:i + size] for i in range(0, len(order), size)]

    def _probe(self, plan: Plan, arm: str, rnd: int, rep: int, mode: str, main: Path, wd: Path,
               base_env: dict, comp: D.CompactResult | None, cut: bool | None) -> None:
        facts = {f.uid: f for f in plan.facts}
        hidden = tuple(self.sc.get("probe_hidden_tools", ()))
        lookup_arm = bool(self.cfg["arms"][arm]["skill"])
        dumps = D.dumps_dir(self.cfg, self.exp, arm, plan.session_id)
        fork_mode = self.sc.get("probe_fork", "pruned")
        if fork_mode not in ("pruned", "full"):
            raise ValueError(f"probe_fork must be 'pruned' or 'full', not {fork_mode!r}")
        tools = self.sc.get("probe_builtin_tools", True)
        if fork_mode == "full" and tools:
            raise ValueError("probe_fork 'full' needs probe_builtin_tools false: with file tools the agent could read the history")
        for k, uids in enumerate(self._probe_sets(plan, mode, rnd)):
            questions = "\n".join(f"Q{i + 1}: {facts[u].question}" for i, u in enumerate(uids))
            name = f"r{rnd}_rep{rep}_{mode}_{k:02d}"
            fork = D.prepare_fork(self.cfg, main, main.parent / "probes" / f"{name}.jsonl",
                                  prune=fork_mode == "pruned")
            # The probe's own copy of the line's out-of-session state (private
            # HOME; pi-lcm's store), so probe traffic never reaches the main line.
            env = {**base_env, **D.copy_probe_state(self.cfg, self.exp, arm, plan.session_id, name)}
            n_fork = S.line_count(fork)
            tr = D.turn(self.cfg, fork, wd, env, self.probe_tpl.format(questions=questions), builtin_tools=tools,
                        hidden_tools=hidden)
            answers = parse_answers(tr.text)
            followup, kind = False, "none"
            missing = [i + 1 for i in range(len(uids)) if i + 1 not in answers]
            if tr.ok and (not answers or (missing and self.sc.get("partial_followup"))):
                # No answer lines (e.g. the model wrote tool calls as text for tools it
                # does not have), or (v2) some questions unanswered: one fixed follow-up
                # on the same fork, same for every arm.
                followup, kind = True, "no_answers" if not answers else "partial"
                msg = PROBE_FOLLOWUP if not answers else PARTIAL_FOLLOWUP.format(
                    missing=", ".join(f"Q{n}" for n in missing))
                t2 = D.turn(self.cfg, fork, wd, env, msg, builtin_tools=tools, hidden_tools=hidden)
                answers = {**parse_answers(t2.text), **answers}
                tr = D.TurnResult(t2.ok, t2.error, tr.text + "\n\n[follow-up]\n" + t2.text, tr.calls + t2.calls,
                                  tr.tokens_in + t2.tokens_in, tr.tokens_out + t2.tokens_out, tr.duration_ms + t2.duration_ms,
                                  usageproxy.merge(tr.usage, t2.usage))
            names = [c["name"] for c in tr.calls]
            # Extension (recovery) tools: context_lookup, lcm_*, pi_fold_context.
            recovery = [n for n in names if n not in D.BUILTIN_FILE_TOOLS]
            # Dumps are allowed only with pruned forks (the pre-session-lookup design);
            # on full copies the lookup reads the session in-process, not via the agent.
            leak = D.audit_leak(tr.calls, wd, allowed=(dumps,) if lookup_arm and fork_mode == "pruned" else ())
            # A0 must never be compacted: pi's auto-compaction is disabled for it,
            # and any compaction entry in its fork flags the trial.
            a0_compacted = (not self.cfg["arms"][arm]["compact"]) and any(
                e.get("type") == "compaction" for e in S.entries_after(fork, n_fork))
            for i, u in enumerate(uids):
                f = facts[u]
                _append(self.trials_path, {
                    "experiment": self.exp, "session_id": plan.session_id, "arm": arm, "round": rnd,
                    "rep": rep, "probe_mode": mode, "fact_uid": u, "source": f.source,
                    "position": f.position, "kind": f.kind, "is_lure": f.is_lure,
                    "question": f.question, "value": f.value, "value_old": f.value_old,
                    "accept": f.accept, "intrusion": f.intrusion,
                    "question_number": i + 1, "answer": answers.get(i + 1), "raw_text": tr.text,
                    "turn_ok": tr.ok, "turn_error": tr.error,
                    "lookup_used": "context_lookup" in names, "reread_used": any(n in ("read", "bash") for n in names),
                    "recovery_tool_calls": len(recovery), "recovery_tools": sorted(set(recovery)),
                    "extensions": D.arm_extensions(self.cfg, arm),
                    "n_tool_calls": len(names), "probe_followup": followup, "tokens_in": tr.tokens_in, "tokens_out": tr.tokens_out,
                    "probe_ms": tr.duration_ms, "batch_size": len(uids), "batch_index": k,
                    # every model call of this probe batch (subagents included), from the
                    # usage proxy; the same value on each row of the batch.
                    "usage": tr.usage,
                    "round_planted": f.round_planted, "followup_kind": kind, "leak": leak,
                    "a0_compacted": a0_compacted, "arm_env": D.arm_env_record(env),
                    "summary_chars": comp.summary_chars if comp else None, "cut_ok": cut,
                    "ts": datetime.now(timezone.utc).isoformat(), **self.meta, **self.row_extra,
                })

    def _compact(self, plan: Plan, arm: str, rnd: int, main: Path, wd: Path, env: dict,
                 fact_ids: dict[str, list[str]]) -> tuple[D.CompactResult, bool | None]:
        real_before = D.real_context_tokens(self.cfg, main)
        comp = D.compact(self.cfg, main, wd, env, timeout_s=D.compact_timeout_s(self.cfg, arm))
        early = [e for f in plan.facts if f.position == "early" for e in fact_ids.get(f.uid, [])]
        late = [e for f in plan.facts if f.position == "late" for e in fact_ids.get(f.uid, [])]
        cut = D.cut_ok(main, comp.first_kept_id, early, late) if comp.ok else None
        extra = dict(self.row_extra)
        if self.cont:
            # Study 1b: what the checkpoint still carries under mech's size guard
            # (None for pi's summary, whose size is summary_chars).
            ents = S.read_entries(main)
            last = S.last_compaction(ents)
            extra["mech_checkpoint"] = S.mech_checkpoint_stats(ents, last) if comp.ok and last else None
        _append(self.comp_path, {
            "experiment": self.exp, "session_id": plan.session_id, "arm": arm, "round": rnd,
            "ok": comp.ok, "error": comp.error, "summary_chars": comp.summary_chars,
            "tokens_before": comp.tokens_before, "real_tokens_before": real_before,
            "real_tokens_after": D.real_context_tokens(self.cfg, main) if comp.ok else None,
            "first_kept_id": comp.first_kept_id,
            "details_kind": comp.details_kind, "compact_ms": comp.duration_ms, "cut_ok": cut,
            **self._treatment(arm, comp.compactor if comp.ok else None),
            "details": comp.details, "extensions": D.arm_extensions(self.cfg, arm), "usage": comp.usage,
            "ts": datetime.now(timezone.utc).isoformat(), **self.meta, **extra,
        })
        return comp, cut

    def _fold_commit(self, plan: Plan, arm: str, rnd: int, main: Path, wd: Path,
                     env: dict) -> tuple[D.CompactResult, None]:
        """pi-fold's per-round commit turn (driver.fold_commit), recorded as a
        compaction row so analysis treats every arm alike. real_tokens_before is
        the unfolded session context, measured as for the other arms;
        real_tokens_after cannot be: pi-fold's folds exist only in its context
        hook's projection, not in the context pi builds from the file. The
        provider's prompt sizes before and after the fold (including system
        prompt and tools) are in details."""
        real_before = D.real_context_tokens(self.cfg, main)
        comp = (D.billion_commit if "billion" in D.arm_extensions(self.cfg, arm) else D.fold_commit)(self.cfg, main, wd, env)
        d = comp.details or {}
        _append(self.comp_path, {
            "experiment": self.exp, "session_id": plan.session_id, "arm": arm, "round": rnd,
            "ok": comp.ok, "error": comp.error, "summary_chars": comp.summary_chars,
            "tokens_before": comp.tokens_before, "real_tokens_before": real_before,
            "real_tokens_after": None,
            "provider_tokens_before": d.get("provider_tokens_before"),
            "provider_tokens_after": d.get("provider_tokens_after"),
            "first_kept_id": None, "details_kind": comp.details_kind, "compact_ms": comp.duration_ms,
            "cut_ok": None, **self._treatment(arm, comp.compactor if comp.ok else None),
            "details": d, "extensions": D.arm_extensions(self.cfg, arm), "commit_turn": True, "usage": comp.usage,
            "fold_method": self.cfg["fold"].get("commit", "turn"),
            "ts": datetime.now(timezone.utc).isoformat(), **self.meta,
        })
        return comp, None

    def _reduce(self, plan: Plan, arm: str, rnd: int, main: Path, wd: Path, env: dict,
                fact_ids: dict[str, list[str]]) -> tuple[D.CompactResult | None, bool | None]:
        """The arm's context reduction at the end of a round: pi compaction (any
        compactor behind it), pi-fold's commit turn, or nothing (A0)."""
        a = self.cfg["arms"][arm]
        if a["compact"]:
            return self._compact(plan, arm, rnd, main, wd, env, fact_ids)
        if a.get("commit_turn"):
            return self._fold_commit(plan, arm, rnd, main, wd, env)
        return None, None

    def _treatment(self, arm: str, found: str | None) -> dict:
        """Treatment check: which compactor actually produced the row (read from
        the session file) against the arm's. A mismatch (e.g. pi-lcm returned
        nothing and pi's own summary ran) is data, not retried; analysis
        excludes or reports it."""
        want = D.expected_compactor(self.cfg, arm)
        return {"compactor": found, "compactor_expected": want,
                "treatment_ok": None if want is None or found is None else found == want}

    def _modes_and_reps(self, plan: Plan) -> list[tuple[str, int]]:
        run = self.cfg["run"]
        rng = random.Random(plan.seed + 7)
        out = [(run["probe_mode"], 0)]
        if run["probe_mode"] == "batched" and rng.random() < self.sc.get("single_mode_subset", run["single_mode_subset"]):
            out.append(("single", 0))
        if run["replicates"] > 1 and rng.random() < run["replicate_subset"]:
            out += [(run["probe_mode"], r) for r in range(1, run["replicates"])]
        return out

    # ------------------------------------------------------------------ main
    def run(self, arms: list[str] | None = None, indices: list[int] | None = None, log=print) -> bool:
        """Run every (session, arm) line still missing trials. Returns False if it
        stopped early because the model may not be used now (see schedule.py)."""
        arms = arms or self.sc["arms"]
        proxy = None
        if self.cfg.raw.get("usage_proxy", {}).get("enabled"):
            # Every model call, whoever makes it, goes through the proxy (usageproxy.py).
            proxy = usageproxy.UsageProxy(self.cfg["usage_proxy"]["upstream"]).start()
            D.set_usage_proxy(self.cfg, proxy, self.cfg.path(self.cfg["run"]["runs_dir"]) / self.exp
                              / ".usage-proxy-models.json")
        try:
            self._run(arms, indices, log)
        except _Stop as e:
            log(f"stopping before the next line: {e}")
            return False
        finally:
            if proxy is not None:
                D.set_usage_proxy(self.cfg, None)
                proxy.stop()
        return True

    def _gate(self) -> None:
        # Checked before a line starts: a started line runs to the end.
        ok, why = schedule.llm_allowed(self.cfg)
        if not ok:
            raise _Stop(why)

    def _run(self, arms: list[str], indices: list[int] | None, log) -> None:
        for plan in self.plans(indices):
            for arm in arms:
                try:
                    env = D.arm_env(self.cfg, arm, self.exp, plan.session_id,
                                    overrides=self.sc.get("arm_env", {}).get(arm))
                except ValueError as e:
                    log(f"skip {arm}: {e}")
                    continue
                if self.exp == "E2":
                    self._run_e2(plan, arm, env, log)
                elif self.cont:
                    self._run_continue(plan, arm, env, log)
                else:
                    self._run_e1(plan, arm, env, log)

    def _fresh_main(self, plan: Plan, arm: str) -> tuple[Path, Path]:
        # wd: the agent's (empty) project dir; main: the session line, kept in a
        # sibling directory the agent cannot read (see driver.sessions_dir).
        wd = D.setup_workdir(self.cfg, self.exp, arm, plan.session_id)
        main = D.sessions_dir(self.cfg, self.exp, arm, plan.session_id) / "main.jsonl"
        return wd, main

    def _run_e2(self, plan: Plan, arm: str, env: dict, log) -> None:
        todo = [(m, r) for m, r in self._modes_and_reps(plan) if (plan.session_id, arm, 1, r, m) not in self.done]
        if not todo:
            return
        self._gate()
        wd, main = self._fresh_main(plan, arm)
        names = ["plant", "bulk", "tail"]
        if arm == "none" and self._est_tokens(plan, names) > 0.9 * self.cfg.context_window:
            log(f"{plan.session_id}/{arm}: skipped, session exceeds context window")
            return
        fact_ids = self._write_phases(plan, main, wd, names)
        comp, cut = self._reduce(plan, arm, 1, main, wd, env, fact_ids)
        if comp is not None and not comp.ok:
            log(f"{plan.session_id}/{arm}: compaction failed: {comp.error}")
        for mode, rep in todo:
            self._probe(plan, arm, 1, rep, mode, main, wd, env, comp, cut)
        log(f"{plan.session_id}/{arm}: done")

    def _run_e1(self, plan: Plan, arm: str, env: dict, log) -> None:
        rounds = list(range(1, self.sc["rounds"] + 1))
        if arm == "none":
            rounds = [r for r in rounds if r in self.sc.get("a0_rounds", [])]
        modes = self._modes_and_reps(plan)
        if all((plan.session_id, arm, r, rep, m) in self.done for r in rounds for m, rep in modes):
            return
        if (plan.session_id, arm) in self.stopped:
            log(f"{plan.session_id}/{arm}: stopped earlier by a failed compaction; not retried")
            return
        self._gate()
        # Rebuild the main line from scratch: compaction is not resumable mid-line.
        wd, main = self._fresh_main(plan, arm)
        fact_ids = self._write_phases(plan, main, wd, ["plant"])
        for r in rounds if arm != "none" else range(1, max(rounds, default=0) + 1):
            fact_ids.update(self._write_phases(plan, main, wd, [f"round{r}"]))
            comp, cut = self._reduce(plan, arm, r, main, wd, env, fact_ids)
            if comp is not None and not comp.ok:
                log(f"{plan.session_id}/{arm} r{r}: compaction failed: {comp.error}; stopping this line")
                return
            if r not in rounds:
                continue
            if arm == "none":
                names = ["plant"] + [f"round{i}" for i in range(1, r + 1)]
                if self._est_tokens(plan, names) > 0.9 * self.cfg.context_window:
                    log(f"{plan.session_id}/{arm} r{r}: skipped, exceeds context window")
                    continue
            for mode, rep in modes:
                if (plan.session_id, arm, r, rep, mode) not in self.done:
                    self._probe(plan, arm, r, rep, mode, main, wd, env, comp, cut)
            log(f"{plan.session_id}/{arm} r{r}: done")

    # ------------------------------------------------------------------ continuation (Study 1b)
    def _source_line(self, plan: Plan, arm: str) -> tuple[Path | None, dict | None, str | None]:
        """The source experiment's line to continue and its round-`after`
        compaction row, or why it cannot be continued (missing or incomplete:
        no compaction row for every round 1..after, a failed compaction or
        treatment, or round-`after` trials missing). Reads only."""
        src_exp, after = self.cont["experiment"], int(self.cont["after_round"])
        sid = plan.session_id
        src_main = D.sessions_dir(self.cfg, src_exp, arm, sid) / "main.jsonl"
        if not src_main.is_file():
            return None, None, f"no {src_exp} line at {src_main}"
        src_res = D.results_dir(self.cfg, src_exp)
        comps = [c for c in _read_jsonl(src_res / "compactions.jsonl") if c["session_id"] == sid and c["arm"] == arm]
        rounds = sorted(c["round"] for c in comps)
        if rounds != list(range(1, after + 1)):
            return None, None, f"{src_exp} has compaction rows for rounds {rounds}, needs exactly 1..{after}"
        bad = [c["round"] for c in comps if not c["ok"] or c.get("treatment_ok") is False]
        if bad:
            return None, None, f"{src_exp} compaction failed or had the wrong treatment in rounds {bad}"
        want = {f.uid for f in plan.facts if f.round_planted <= after}
        got = {t["fact_uid"] for t in _read_jsonl(src_res / "trials.jsonl")
               if t["session_id"] == sid and t["arm"] == arm and t["round"] == after}
        if not want <= got:
            return None, None, f"{src_exp} has {len(want & got)}/{len(want)} round-{after} trials"
        return src_main, next(c for c in comps if c["round"] == after), None

    def _verify_copy(self, plan: Plan, main: Path, row: dict, scratch: Path) -> tuple[dict[str, list[str]], str | None]:
        """Check the copied line before continuing it:
        - it holds exactly `after` compactions and ends with the last one, which
          matches the recorded round-`after` row (first kept entry, summary
          size, compactor);
        - its header, plant phase and rounds 1..after are this plan's phases as
          SessionWriter writes them (replayed into `scratch`, each round onto
          the copied prefix before it), each round followed by its compaction
          (pi's session-state entries, S.SESSION_STATE_TYPES, may come between).
        Returns the planted facts' entry ids (for cut_ok) and a problem, if any."""
        after = int(self.cont["after_round"])
        ents = S.read_entries(main)
        comp_at = [i for i, e in enumerate(ents) if e.get("type") == "compaction"]
        if len(comp_at) != after or comp_at[-1] != len(ents) - 1:
            return {}, f"the line has {len(comp_at)} compactions and must end with compaction {after}"
        last = ents[-1]
        if (last.get("firstKeptEntryId") != row["first_kept_id"]
                or len(last.get("summary") or "") != row["summary_chars"]
                or (row.get("compactor") is not None and S.compactor_of(last) != row["compactor"])):
            return {}, f"its last compaction does not match the recorded round-{after} row"
        cwd = Path(ents[0].get("cwd") or ".")  # the source's header, so the replayed header is comparable
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.unlink(missing_ok=True)
        fact_ids = self._write_phases(plan, scratch, cwd, ["plant"])
        pos = S.line_count(scratch)
        if S.read_entries(scratch) != ents[:pos]:
            return {}, "its header or plant phase differs from the plan"
        for r in range(1, after + 1):
            with scratch.open("w", encoding="utf-8") as f:
                for e in ents[:pos]:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            self._write_phases(plan, scratch, cwd, [f"round{r}"])
            new = S.read_entries(scratch)[pos:]
            if ents[pos:pos + len(new)] != new:
                return {}, f"its round{r} entries differ from the plan"
            pos += len(new)
            while pos < comp_at[r - 1] and ents[pos].get("type") in S.SESSION_STATE_TYPES:
                pos += 1  # e.g. pi's thinking_level_change before the first compaction
            if pos != comp_at[r - 1]:
                return {}, f"round{r} is not followed by compaction {r}"
            pos += 1
        scratch.unlink()
        return fact_ids, None

    def _run_continue(self, plan: Plan, arm: str, env: dict, log) -> None:
        """Continue the source experiment's line from its round-`after`
        compaction (PROTOCOL.md, Study 1b): copy the main line and the line's
        out-of-session state into this experiment's tree, verify the copy, then
        for each later round append the round's work, compact and probe as
        _run_e1 does. The source tree is only read. A missing or incomplete
        source line is skipped with its reason and nothing is recorded; a line
        with trials missing is redone from a fresh copy (compaction is not
        resumable), probing only what is missing."""
        src_exp, after = self.cont["experiment"], int(self.cont["after_round"])
        tag = f"{plan.session_id}/{arm}"
        if not self.cfg["arms"][arm]["compact"]:
            log(f"{tag}: skipped, {arm} does not compact (not part of a continuation study)")
            return
        rounds = list(range(after + 1, self.sc["rounds"] + 1))
        modes = self._modes_and_reps(plan)
        if all((plan.session_id, arm, r, rep, m) in self.done for r in rounds for m, rep in modes):
            return
        if (plan.session_id, arm) in self.stopped:
            log(f"{tag}: stopped earlier by a failed compaction; not retried")
            return
        src, row, why = self._source_line(plan, arm)
        if why:
            log(f"{tag}: skipped, {why}")
            return
        src_root = (self.cfg.path(self.cfg["run"]["runs_dir"]) / src_exp).resolve()
        dst_lines = D.lines_dir(self.cfg, self.exp, arm, plan.session_id)
        for d in (dst_lines, D.workdir(self.cfg, self.exp, arm, plan.session_id)):
            if src_root == d.resolve() or src_root in d.resolve().parents:
                raise RuntimeError(f"{self.exp} must not write into {src_root}")
        self._gate()
        wd, main = self._fresh_main(plan, arm)
        digest = _sha256(src)
        shutil.copyfile(src, main)
        if _sha256(main) != digest:
            raise RuntimeError(f"{tag}: the copy of {src} does not match its hash")
        src_lines = D.lines_dir(self.cfg, src_exp, arm, plan.session_id)
        for name in ("home", "dumps", "lcm"):  # the line's out-of-session state, where it has any
            if (src_lines / name).is_dir():
                shutil.copytree(src_lines / name, dst_lines / name, symlinks=True, dirs_exist_ok=True)
        # The same treatment as the source line: identical project files
        # (.pi: extension copies, settings) in the agent's working directory.
        diff = _tree_diff(D.agent_cwd(self.cfg, src_exp, arm, plan.session_id) / ".pi", wd / ".pi")
        if diff:
            log(f"{tag}: skipped, the arm's project files differ from the {src_exp} line: {diff[:5]}")
            return
        fact_ids, why = self._verify_copy(plan, main, row, dst_lines / "verify" / "replay.jsonl")
        if why:
            log(f"{tag}: skipped, the copied {src_exp} line fails verification: {why}")
            return
        self.row_extra = {"source_experiment": src_exp, "source_session": plan.session_id,
                          "source_sha256": digest, "continued_after_round": after}
        try:
            for r in rounds:
                fact_ids.update(self._write_phases(plan, main, wd, [f"round{r}"]))
                comp, cut = self._reduce(plan, arm, r, main, wd, env, fact_ids)
                if comp is not None and not comp.ok:
                    log(f"{tag} r{r}: compaction failed: {comp.error}; stopping this line")
                    return
                for mode, rep in modes:
                    if (plan.session_id, arm, r, rep, mode) not in self.done:
                        self._probe(plan, arm, r, rep, mode, main, wd, env, comp, cut)
                log(f"{tag} r{r}: done")
        finally:
            self.row_extra = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_diff(a: Path, b: Path) -> list[str]:
    """Relative paths whose presence or content differs between two trees."""
    def files(d: Path) -> dict[str, Path]:
        return {p.relative_to(d).as_posix(): p for p in d.rglob("*") if p.is_file()} if d.is_dir() else {}
    fa, fb = files(a), files(b)
    if not fa:
        return [f"{a} is missing or empty"]
    return sorted(k for k in fa.keys() | fb.keys()
                  if k not in fa or k not in fb or fa[k].read_bytes() != fb[k].read_bytes())
