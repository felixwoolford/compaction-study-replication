"""Run Study 2 (PROTOCOL.md v2 §6.3): continuation after a compaction mid-task. Resumable.

Per (session, arm): write the synthetic history into the main line, compact
(except arms with compact=false), prepare the pruned fork, put the repository
at state k into the agent's working directory, send the continuation message
and up to `max_followups` fixed follow-ups while the final checklist has open
steps (or none can be parsed), then run the checkers and score.

Outputs (append-only JSONL):
  results/<exp>/episodes.jsonl     one row per (session, arm)
  results/<exp>/compactions.jsonl  one row per compaction
  runs/<exp>/plans/<sid>.json      the exact plan used
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from . import driver as D
from . import pisession as S
from . import schedule
from .config import Config
from .continuation import TaskPlan, build_task_plan, write_repo
from .runner import _append, _git_commit, _read_jsonl
from .scoring_s2 import parse_checklist, score_episode


audit_view = D.audit_view  # moved to driver.py; kept for callers


class _Stop(Exception):
    """The model may not be used now; stop before the next line."""


class S2Runner:
    def __init__(self, cfg: Config, scenario: dict):
        self.cfg = cfg
        self.sc = scenario
        self.exp = scenario["experiment"]
        self.res_dir = D.results_dir(cfg, self.exp)
        self.runs_dir = cfg.path(cfg["run"]["runs_dir"]) / self.exp
        self.ep_path = self.res_dir / "episodes.jsonl"
        self.comp_path = self.res_dir / "compactions.jsonl"
        self.meta = {"git_commit": _git_commit(cfg.root), "pi_version": cfg["pi"]["version"],
                     "model": f"{cfg['model']['provider']}/{cfg['model']['id']}"}
        self.done = {(r["session_id"], r["arm"]) for r in _read_jsonl(self.ep_path)}

    def plans(self, indices: list[int] | None = None) -> list[TaskPlan]:
        out = []
        for i in indices if indices is not None else range(self.sc["n_sessions"]):
            p = build_task_plan(self.sc, i)
            path = self.runs_dir / "plans" / f"{p.session_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(p.to_dict(), indent=1), encoding="utf-8")
            out.append(p)
        return out

    def run(self, arms: list[str] | None = None, indices: list[int] | None = None, log=print) -> bool:
        """Run every (session, arm) without a row yet. False if stopped by the schedule gate."""
        arms = arms or self.sc["arms"]
        keep = int(self.sc.get("keep_recent_tokens", self.cfg.keep_recent_tokens))
        if keep != self.cfg.keep_recent_tokens:
            log(f"warning: scenario keep_recent_tokens {keep} != config {self.cfg.keep_recent_tokens}")
        try:
            for plan in self.plans(indices):
                for arm in arms:
                    if (plan.session_id, arm) in self.done:
                        continue
                    try:
                        env = D.arm_env(self.cfg, arm, self.exp, plan.session_id,
                                        overrides=self.sc.get("arm_env", {}).get(arm))
                    except ValueError as e:
                        log(f"skip {arm}: {e}")
                        continue
                    self._gate()
                    self._episode(plan, arm, env, log)
                    self.done.add((plan.session_id, arm))
        except _Stop as e:
            log(f"stopping before the next line: {e}")
            return False
        return True

    def _gate(self) -> None:
        ok, why = schedule.llm_allowed(self.cfg)
        if not ok:
            raise _Stop(why)

    def _write_history(self, plan: TaskPlan, main, wd) -> list[str]:
        """Write the history phases; return the entry ids of the step work (core)."""
        core_ids: list[str] = []
        for idx, ph in enumerate(plan.phases):
            w = S.SessionWriter(main, cwd=str(wd), version=int(self.cfg["pi"]["session_version"]),
                                provider=self.cfg["model"]["provider"], model=self.cfg["model"]["id"],
                                seed=plan.seed * 100 + idx, thinking_level=self.cfg["model"]["thinking_level"])
            for st in ph["steps"]:
                ids = w.step(st)
                if st.get("tag") == "core":
                    core_ids.extend(ids)
        return core_ids

    def _row(self, plan: TaskPlan, arm: str, env: dict, **fields) -> dict:
        return {"experiment": self.exp, "session_id": plan.session_id, "task": plan.task, "arm": arm,
                "k": plan.k, "false_step": plan.false_step, "n_steps": plan.n_steps, **fields,
                "ts": datetime.now(timezone.utc).isoformat(), **self.meta, "arm_env": D.arm_env_record(env)}

    def _episode(self, plan: TaskPlan, arm: str, env: dict, log) -> None:
        cfg, sid = self.cfg, plan.session_id
        wd = D.setup_workdir(cfg, self.exp, arm, sid)
        sdir = D.sessions_dir(cfg, self.exp, arm, sid)
        main = sdir / "main.jsonl"
        core_ids = self._write_history(plan, main, wd)
        write_repo(plan.repo_state(), wd)
        compacts = bool(cfg["arms"][arm]["compact"])
        comp, cut = None, None
        if compacts:
            comp = D.compact(cfg, main, wd, env)
            cut = D.cut_ok(main, comp.first_kept_id, core_ids, []) if comp.ok else None
            _append(self.comp_path, {
                "experiment": self.exp, "session_id": sid, "arm": arm, "round": 1, "ok": comp.ok,
                "error": comp.error, "summary_chars": comp.summary_chars, "tokens_before": comp.tokens_before,
                "first_kept_id": comp.first_kept_id, "details_kind": comp.details_kind,
                "compact_ms": comp.duration_ms, "cut_ok": cut,
                "ts": datetime.now(timezone.utc).isoformat(), **self.meta})
            if not comp.ok:
                log(f"{sid}/{arm}: compaction failed: {comp.error}")
                _append(self.ep_path, self._row(
                    plan, arm, env, turns=0, followups=0, turn_ok=False, turn_error=f"compaction failed: {comp.error}",
                    timeout=False, compact_ok=False, final_checklist=None, checker=None, fabricated_progress=None,
                    inherited_false_claim=None, unsupported_reports=None, verified_false_step=None, success=None,
                    repeated_work=None, needs_review=True, leak=False, a0_compacted=False, n_tool_calls=0,
                    tokens_in=0, tokens_out=0, duration_ms=0, summary_chars=comp.summary_chars, cut_ok=None))
                return
        fork = D.prepare_fork(cfg, main, sdir / "continuation.jsonl")
        n_fork = S.line_count(fork)

        results = [D.turn(cfg, fork, wd, env, plan.continue_message, builtin_tools=True)]
        followups = 0
        while results[-1].ok and followups < plan.max_followups:
            cl = parse_checklist(results[-1].text, plan.n_steps)
            if cl is not None and all(cl.values()):
                break
            results.append(D.turn(cfg, fork, wd, env, plan.followup_message, builtin_tools=True))
            followups += 1

        checker = plan.check(wd)
        turns = [{"text": r.text, "calls": r.calls} for r in results]
        scores = score_episode([{"n": s["n"], "files": s["files"]} for s in plan.steps], plan.k, plan.false_step,
                               turns, checker, wd=str(wd))
        calls = [c for r in results for c in r.calls]
        allowed = (D.dumps_dir(cfg, self.exp, arm, sid),) if cfg["arms"][arm]["skill"] else ()
        leak = D.audit_leak(audit_view(calls), wd, allowed=allowed)
        a0_compacted = (not compacts) and any(e.get("type") == "compaction" for e in S.entries_after(fork, n_fork))
        err = next((r.error for r in results if r.error), None)
        _append(self.ep_path, self._row(
            plan, arm, env, turns=len(results), followups=followups, turn_ok=all(r.ok for r in results),
            turn_error=err, timeout=bool(err and "timeout" in err.lower()), compact_ok=True if compacts else None,
            **scores, leak=leak, a0_compacted=a0_compacted, n_tool_calls=len(calls),
            tokens_in=sum(r.tokens_in for r in results), tokens_out=sum(r.tokens_out for r in results),
            duration_ms=sum(r.duration_ms for r in results), summary_chars=comp.summary_chars if comp else None,
            cut_ok=cut, turn_texts=[r.text for r in results]))
        log(f"{sid}/{arm}: done (success={scores['success']}, fabricated={scores['fabricated_progress']})")
