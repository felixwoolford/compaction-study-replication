#!/usr/bin/env node
// pi-bridge: the ONLY component that makes pi do something.
//
//   node pi-bridge.mjs turn     < {"sessionFile", "cwd", "message", "timeoutMs", "followUps"?}
//   node pi-bridge.mjs compact  < {"sessionFile", "cwd", "timeoutMs"}
//   node pi-bridge.mjs context  < {"sessionFile", "timeoutMs"}
//   node pi-bridge.mjs command  < {"sessionFile", "cwd", "command", "timeoutMs"}   (extension slash command, no model call)
//   node pi-bridge.mjs continue < {"sessionFile", "cwd", "timeoutMs", "noBuiltinTools"?}
//
// `continue` resumes a session whose last entry is a tool result (pi's
// agent.continue()): the agent carries on from that point, with no new message.
//
// `turn` appends one literal user message and runs to completion (then each
// of `followUps`, in the same process: pi-fold's commit turn needs a measured
// reply and then a second context pass in one process); `compact`
// appends one compaction entry. `context` makes no model call: it returns a
// hash of the context pi would send (SessionManager.buildSessionContext), used
// to show a pruned fork yields the same context as the full file.
// Measurements belong to driver.py, not here.
// Output: exactly one JSON line: {"ok": true} or {"ok": false, "error": "..."}.
//
// Uses pi's SDK: print mode has --session, but /compact is a TUI command,
// not a headless prompt command (pi 1.0.0). SessionManager.open writes directly
// to the supplied file, including outside pi's sessions directory; no copy-back.
// The harness environment reaches extensions unchanged. CEVAL_MODEL_* selects
// the study model/thinking explicitly, independent of session/global defaults.
// Exactly the arm's extensions are loaded (CEVAL_EXTENSIONS: a JSON array of
// absolute paths, possibly empty), plus the arm-local skill and pi's llama.cpp
// provider when needed. Global skills/extensions and ancestor context files are excluded.
// Retry overrides are in-memory only: failed operations are data, not retried.
// Model config (incl. per-request samplingParams) comes from the study's own
// models file (CEVAL_PI_MODELS_FILE, committed in the repo), not the agent-dir
// models.json; credentials and the catalog store still come from the agent dir.
// pi 1.0.3's llama.cpp provider discovers models only from a router server; the
// study server runs a single model, so the models file declares it explicitly.

import { existsSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Extensions may log diagnostics, but stdout is reserved for the response.
console.log = console.info = console.debug = console.error;
let session;
let finished = false;

function finish(result) {
  if (finished) return;
  finished = true;
  // Exit only once the reply is flushed: a large reply (context text) would be
  // truncated on a pipe by an immediate process.exit.
  process.stdout.write(JSON.stringify(result) + "\n", () => process.exit(result.ok ? 0 : 1));
}

async function stop(error) {
  // Abort cooperatively so pi can persist an interrupted response. Bound cleanup
  // too: an unresponsive provider/tool must not keep the bridge alive indefinitely.
  const forcedExit = setTimeout(() => finish({ ok: false, error }), 1000);
  try {
    session?.abortCompaction();
    await session?.abort();
    session?.dispose();
  } finally {
    clearTimeout(forcedExit);
    finish({ ok: false, error });
  }
}

function requiredEnv(name) {
  const value = process.env[name];
  if (!value) throw new Error(`missing environment variable ${name}`);
  return value;
}

function armExtensions() {
  // The harness always says which extensions the arm loads; a missing list is a
  // harness bug, not "load none".
  let list;
  try {
    list = JSON.parse(requiredEnv("CEVAL_EXTENSIONS"));
  } catch (e) {
    throw new Error(`invalid CEVAL_EXTENSIONS: ${e.message}`);
  }
  if (!Array.isArray(list) || !list.every((p) => typeof p === "string" && isAbsolute(p))) {
    throw new Error("CEVAL_EXTENSIONS must be a JSON array of absolute paths");
  }
  for (const p of list) if (!existsSync(p)) throw new Error(`arm extension not found: ${p}`);
  return list;
}

async function openSession({ sessionFile, cwd, noBuiltinTools, hiddenTools }) {
  const packageDir = requiredEnv("PI_PACKAGE_DIR");
  const sdk = await import(pathToFileURL(join(packageDir, "dist/index.js")).href);
  const provider = requiredEnv("CEVAL_MODEL_PROVIDER");
  const modelId = requiredEnv("CEVAL_MODEL_ID");
  const thinkingLevel = requiredEnv("CEVAL_MODEL_THINKING_LEVEL");
  const agentDir = requiredEnv("CEVAL_PI_AGENT_DIR");
  const modelsFile = resolve(requiredEnv("CEVAL_PI_MODELS_FILE"));
  if (!existsSync(modelsFile)) throw new Error(`models file not found: ${modelsFile}`);
  cwd = resolve(cwd);
  sessionFile = resolve(sessionFile);
  if (!existsSync(sessionFile)) throw new Error(`session file not found: ${sessionFile}`);
  const extensions = armExtensions();
  const skill = join(cwd, ".pi/skills/context-dump");
  const extensionFactories = [];
  const additionalExtensionPaths = [...extensions];
  if (provider === "llama.cpp") {
    // SDK sessions don't install CLI built-ins. Reuse pi's actual provider,
    // including its stored credentials/catalog and models.json overrides.
    const { default: factory } = await import(
      pathToFileURL(join(packageDir, "dist/extensions/llama/index.js")).href
    );
    extensionFactories.push({ name: "llama.cpp", factory, builtin: true });
    additionalExtensionPaths.unshift("builtin:llama.cpp");
  }
  const settingsManager = sdk.SettingsManager.create(cwd, agentDir);
  settingsManager.applyOverrides({ retry: { enabled: false, maxRetries: 0, provider: { maxRetries: 0 } } });
  const modelRuntime = await sdk.ModelRuntime.create({
    // CEVAL_PI_AUTH_FILE: the harness's own minimal auth file (usage proxy runs):
    // pi takes llama.cpp's address from the provider's auth entry, not the models file.
    authPath: process.env.CEVAL_PI_AUTH_FILE ? resolve(process.env.CEVAL_PI_AUTH_FILE) : join(agentDir, "auth.json"),
    modelsPath: modelsFile,
    modelsStorePath: join(agentDir, "models-store.json"),
  });
  const services = await sdk.createAgentSessionServices({
    cwd, agentDir, settingsManager, modelRuntime,
    resourceLoaderOptions: {
      noExtensions: true, additionalExtensionPaths, extensionFactories,
      noSkills: true, additionalSkillPaths: existsSync(skill) ? [skill] : [],
      noPromptTemplates: true, noThemes: true, noContextFiles: true,
    },
  });
  const loadErrors = services.resourceLoader.getExtensions().errors;
  if (loadErrors.length) throw new Error(`extension load failed: ${JSON.stringify(loadErrors)}`);
  const errors = services.diagnostics.filter((d) => d.type === "error");
  if (errors.length) throw new Error(`pi startup failed: ${JSON.stringify(errors)}`);
  const settingsErrors = settingsManager.drainErrors();
  if (settingsErrors.length) throw new Error(`settings load failed: ${settingsErrors.map((e) => e.error).join("; ")}`);
  const model = services.modelRuntime.getModel(provider, modelId);
  if (!model) throw new Error(`study model unavailable: ${provider}/${modelId}`);
  const created = await sdk.createAgentSessionFromServices({
    services, model, thinkingLevel,
    // Probes: no read/bash/edit/write, so the agent cannot open session files on
    // disk; extension tools (context_lookup) stay.
    ...(noBuiltinTools ? { noTools: "builtin" } : {}),
    sessionManager: sdk.SessionManager.open(sessionFile, undefined, cwd),
  });
  session = created.session;
  if (resolve(session.sessionFile) !== sessionFile) throw new Error("pi opened a different session file");
  if (session.thinkingLevel !== thinkingLevel) throw new Error(`pi changed thinking level to ${session.thinkingLevel}`);
  const extensionErrors = [];
  await session.bindExtensions({ mode: "print", onError: (e) => extensionErrors.push(e.error) });
  return extensionErrors;
}

async function runTurn(req) {
  const errors = await openSession(req);
  if (req.hiddenTools?.length) {
    // Probes hide extension tools that are useless there (dump_context).
    const hide = new Set(req.hiddenTools);
    session.setActiveToolsByName(session.getActiveToolNames().filter((n) => !hide.has(n)));
  }
  // Failure events control success, not measurements of text/tokens. In particular,
  // prompt() can resolve even when the provider returned an error assistant message.
  session.subscribe((event) => {
    if (event.type === "message_end" && event.message.role === "assistant" &&
        ["error", "aborted"].includes(event.message.stopReason)) {
      errors.push(event.message.errorMessage || `Request ${event.message.stopReason}`);
    }
  });
  for (const message of [req.message, ...(req.followUps ?? [])]) {
    await session.prompt(message, { expandPromptTemplates: false });
    await session.waitForIdle();
    if (errors.length) throw new Error(errors.join("; "));
  }
}

async function runContext(req) {
  const { createHash } = await import("node:crypto");
  const sdk = await import(pathToFileURL(join(requiredEnv("PI_PACKAGE_DIR"), "dist/index.js")).href);
  const file = resolve(req.sessionFile);
  if (!existsSync(file)) throw new Error(`session file not found: ${file}`);
  const ctx = sdk.SessionManager.open(file).buildSessionContext();
  const json = JSON.stringify(ctx.messages);
  return { messages: ctx.messages.length, sha256: createHash("sha256").update(json).digest("hex"),
    ...(req.includeText ? { text: json } : {}) };
}

// `command` runs one extension slash command (e.g. pi-fold's "/fold") and
// makes NO model call: the name must be a registered extension command
// (otherwise prompt() would send the text to the model), and any assistant
// message during it is an error.
async function runCommand(req) {
  const errors = await openSession(req);
  const text = req.command;
  if (!text.startsWith("/")) throw new Error("command must start with /");
  const name = text.slice(1).split(" ")[0];
  if (!session._extensionRunner?.getCommand?.(name)) throw new Error(`not an extension command: ${name}`);
  session.subscribe((event) => {
    if (event.type === "message_start" && event.message?.role === "assistant") errors.push(`command ${name} started a model call`);
  });
  let outcome;
  await session.prompt(text, { expandPromptTemplates: true, preflightResult: (r) => { outcome = r; } });
  await session.waitForIdle();
  if (outcome !== "handled") errors.push(`command ${name} was not handled by an extension (${outcome})`);
  if (errors.length) throw new Error(errors.join("; "));
}

async function runContinue(req) {
  const errors = await openSession(req);
  session.subscribe((event) => {
    if (event.type === "message_end" && event.message.role === "assistant" &&
        ["error", "aborted"].includes(event.message.stopReason)) {
      errors.push(event.message.errorMessage || `Request ${event.message.stopReason}`);
    }
  });
  await session.agent.continue();
  await session.waitForIdle();
  if (errors.length) throw new Error(errors.join("; "));
}

async function runCompact(req) {
  const errors = await openSession(req);
  await session.compact();
  if (errors.length) throw new Error(errors.join("; "));
}

async function main() {
  const cmd = process.argv[2];
  if (!["turn", "compact", "context", "command", "continue"].includes(cmd)) throw new Error(`unknown command ${cmd}`);
  const chunks = [];
  for await (const c of process.stdin) chunks.push(c);
  const req = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  for (const key of ["sessionFile", ...(cmd === "context" ? [] : ["cwd"]), ...(cmd === "turn" ? ["message"] : []), ...(cmd === "command" ? ["command"] : [])]) {
    if (typeof req[key] !== "string" || !req[key]) throw new Error(`missing or invalid ${key}`);
  }
  if (!Number.isSafeInteger(req.timeoutMs) || req.timeoutMs <= 0) throw new Error("invalid timeoutMs");
  if (req.followUps !== undefined && !(Array.isArray(req.followUps) &&
      req.followUps.every((m) => typeof m === "string" && m))) throw new Error("invalid followUps");
  const timer = setTimeout(() => void stop(`timeout after ${req.timeoutMs} ms`), req.timeoutMs);
  process.once("SIGTERM", () => void stop("terminated by SIGTERM"));
  process.once("SIGINT", () => void stop("terminated by SIGINT"));
  let extra = {};
  try {
    if (cmd === "turn") await runTurn(req);
    else if (cmd === "compact") await runCompact(req);
    else if (cmd === "command") await runCommand(req);
    else if (cmd === "continue") await runContinue(req);
    else extra = await runContext(req);
  } finally {
    clearTimeout(timer);
    session?.dispose();
  }
  finish({ ok: true, ...extra });
}

main().catch((e) => finish({ ok: false, error: String(e?.stack ?? e) }));
