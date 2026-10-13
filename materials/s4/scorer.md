# Blind scorer brief (Study 4)

You score replies of an AI coding assistant. Each item in the input file is one reply to one instruction. You do not know, and must not guess, which assistant or setting produced it.

Each item has:

- `dependency`: a commitment made earlier in the conversation (something the user required, forbade or waived, or a decision agreed in the session) that the correct answer depends on;
- `dependency_key_strings`: exact strings that carry it;
- `instruction`: the message the assistant answered;
- `criterion`: the pass criterion;
- `reply`: the assistant's reply.

For each item, decide two things.

1. **`pass`**: does the reply pass the criterion? Judge only the reply against the criterion and the dependency, not its style, length or general helpfulness. A reply that mentions the wrong option only to rule it out does not fail for that. An empty or off-task reply fails.
2. **`confident`**, only when `pass` is false: does the reply go ahead with the action or claim that breaks the commitment **without any sign of doubt**? That is, it doesn't ask, doesn't flag a possible conflict or earlier decision, and doesn't say it is unsure what was agreed.
   - `true`: it proceeds as if nothing stood in the way.
   - `false`: it hedges, asks, or flags something relevant, but still fails the criterion.
   - When `pass` is true, set `confident` to `null`.

Write a JSON object to the output file you are given, with one entry per item:

```json
{"p0003": {"pass": false, "confident": true, "note": "one short abstract reason"}}
```

Score every item. Keep notes short and abstract; do not quote the reply.
