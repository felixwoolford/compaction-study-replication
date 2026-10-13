# Notices

## Study 4 session data and probes

The 18 sessions Study 4 runs on come from the Hugging Face dataset
[MaxDevv/real-pi-coding-agent-traces-sessions](https://huggingface.co/datasets/MaxDevv/real-pi-coding-agent-traces-sessions),
which collects public pi sessions; each session remains under the licence of its source. All 18 used here come from
MIT-licensed sources (woxQAQ/pi-web; thomasmustier/pi-for-excel-sessions, pi-extensions-sessions, pi-mono-sessions,
pine-of-glass-sessions and others listed in `materials/s4/selection.json`). The sessions are not redistributed here;
`scripts/download_sessions.py` fetches them.

`materials/s4/probes.json` was written for this study. Its instructions and criteria paraphrase, and its
`dependency_text` fields quote short strings from, those MIT-licensed sessions. MIT licence terms of the sources apply
to that quoted material: "Permission is hereby granted, free of charge, to any person obtaining a copy of this
software ... The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software." (see each source repository for its copyright notice).

## Third-party software

verbatim (github.com/felixwoolford/verbatim-compact-pi) and pi-vcc (github.com/monotykamary/pi-vcc, release 0.8.12)
are cloned from their repositories by `scripts/setup.py`; neither is redistributed here. pi (`@earendil-works/pi-coding-agent`) is installed by the user.
