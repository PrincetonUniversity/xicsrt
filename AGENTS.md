# XICSRT

Photon-based scientific raytracer (optical + x-ray Bragg reflection) in pure Python.
Runtime deps: `numpy`, `scipy`, `pillow`, `h5py`. Requires Python >= 3.8.

## Running / verifying

A `pytest` unit test suite lives in `tests/` (there is still **no CI**):

    pip install pytest
    pytest tests/

`testing/` separately contains integrated-test Jupyter notebooks
(`integrated_test_*.ipynb`), which are not pytest tests. To verify broader
changes, also run an example script end-to-end:

    python examples/example_00/example_00.py   # point source + detector, no output files

CLI entrypoints (same raytrace, driven by a config file):

    xicsrt config.json            # console_scripts entrypoint
    python -m xicsrt config.json  # equivalent
    python -m xicsrt --mp config.json --processes N   # multiprocessing

## Core execution model

Everything is driven by a nested **config dict** passed to `xicsrt.raytrace(config)`
(see `xicsrt/xicsrt_raytrace.py`). Config sections: `general`, `sources`, `optics`,
`filters`, `scenario`. Defaults and every documented option live in
`xicsrt/xicsrt_config.py:default_config`.

- `strict_config_check` defaults to **True**: unknown config keys raise an
  exception. When adding an option to an element, add it to that element's
  `default_config()` or the run will fail.
- `raytrace` performs `number_of_runs` runs, each doing `number_of_iter`
  iterations; results are combined. Multiprocessing parallelizes over runs.

## Plugin discovery (critical, easy to get wrong)

`objects/_Dispatcher.py` finds elements by globbing files named **`_Xicsrt*.py`**
in the search paths, then loads a class whose name equals the filename minus the
leading underscore. Config selects an element via `class_name`.

- A new source/optic/filter file MUST be named `_Xicsrt<Name>.py` and contain a
  class `Xicsrt<Name>`, or the dispatcher will not find it.
- Built-in search dirs: `xicsrt/sources`, `xicsrt/optics`, `xicsrt/filters`
  (plus the optional `xicsrt_contrib` package if installed).
- User plugin dirs can be added via `config['general']['pathlist']`; these are
  searched before built-ins.

## Element class conventions

- Optics are composed by **multiple inheritance** of an `Interact*` mixin
  (physics) and a `Shape*` mixin (geometry). Example:
  `class XicsrtOpticSphericalCrystal(InteractCrystal, ShapeSphere)` — the class
  body is often just `pass`. Reuse existing mixins rather than duplicating logic.
- Every element/config class is decorated with `@dochelper`
  (`xicsrt/tools/xicsrt_doc.py`), which auto-appends config options from
  `default_config()` (including inherited ones) into the docstring. Keep docstring
  option descriptions accurate; `help()` output is treated as user documentation.
- All config classes derive from `ConfigObject` (`objects/_ConfigObject.py`);
  lifecycle is `setup() -> check_param() -> initialize()`.

## Module layout

- `xicsrt_raytrace.py` – main `raytrace` / `raytrace_single` orchestration.
- `xicsrt_public.py` – interactive helpers (e.g. `get_element`).
- `xicsrt_config.py` – config defaults, merging, path resolution.
- `xicsrt_io.py` – load/save config, images (`.tif`), results (`.hdf5`).
- `objects/` – framework internals (`_Dispatcher`, `_ConfigObject`,
  `_GeometryObject`, `_RayArray`).
- `optics/`, `sources/`, `filters/` – dispatchable elements.
- `tools/` – math/physics helpers (`xicsrt_bragg`, `xicsrt_math`, `xicsrt_voigt`, ...).
- `util/` – `mir*` utilities (logging via `mirlogging`, hdf5, plotting, profiler).
- `devel/` - tracking of code development and pending features.

## Notebook logbooks

- Dated logbook notebooks referenced by title (e.g. "2026-07-30 - W7-X XICS
  Raytracing - SULI 2026 L. Alston, Part 7 - New acceleration updates") live
  outside this repo, under `/u/npablant/code/notebooks/npablant-2019/logbook/`.
  Other sibling directories under `/u/npablant/code/notebooks/` may also hold
  relevant notebooks if not found there.

## Repo / release workflow

- Two remotes: `bitbucket` (`amicitas/xicsrt`) and `princeton`
  (github `PrincetonUniversity/xicsrt`). Releases are pushed to **both**.
- Docs are Sphinx, built from `doc_source/` on ReadTheDocs; API docs are generated
  from docstrings, so docstring formatting matters.
- Release steps (from `setup.py` header): bump `xicsrt/_version.py`, git tag
  `vX.Y.Z`, push tags, merge into `master` and `stable`, then
  `python setup.py sdist bdist_wheel` and `twine upload`.
- `_version.py` documents major/minor/revision semantics: bump **minor** when the
  public API, config compatibility, or output dict structure changes.

## Core Mandates

- Photon statistics are sacrosanct: XICSRT's core tenet is that photon statistics are exactly correct at every element under the basic assumptions. Any optimization (JAX, vectorization, masking, fixed-capacity arrays) must preserve exact statistics — approximations that alter photon statistics are never acceptable without explicit user approval.
- jaxrt synchronization: `xicsrt/jaxrt/` is a parallel JAX engine that mirrors the numpy engine's physics. When modifying sources, optics, or tools in the numpy engine, consult `devel/jaxrt_sync.md` and either mirror the change in `xicsrt/jaxrt/` or record it in that file's divergence log.
- No backwards compatibility (mandatory): This project explicitly does NOT maintain backwards API compatibility. Never add deprecated aliases, shim parameters, optional "legacy" arguments, `**kwargs` passthroughs, or compatibility branches to preserve old call sites. When an API changes, change the signature cleanly and update every caller (source, tests, notebooks, docs, benchmarks) in the same change. Do not warn-and-ignore removed parameters — remove them. Prioritize clean, modern refactoring and optimal API design over compatibility with older versions or structures. If preserving compatibility ever seems necessary, stop and ask the user first rather than adding a shim.
- Input containers (dictionaries or dataclasses): Prefer using a single `options` or `config` container (either a standard dictionary or a `dataclass(frozen=True)`) for function inputs instead of long lists of positional or keyword arguments.
- AI usage disclaimer: At the top of every python source file that is modified by an AI tool, add a disclaimer saying 'This file includes AI generated code'.
  - Add the high-level AI tool identifier eg. Gemini, Claude, Codex etc.
  - Include the specific agent version if possible.
  - Example: `This file includes AI generated code using Claude (Opus 4.8, Sonnet 4.6)`
  - When new functions or methods are *entirely* generated through an AI tool, put a similar appropriate disclaimer in the function/method doc string. Example `This function was AI generated using Claude (Opus 4.8, Sonnet 4.6)`.
- Plan before coding: Before generating or modifying code, always develop a plan outlining the proposed changes and present it to the user for approval. Do not proceed with implementation until the user has reviewed and approved the plan.
- Commit messages (MANDATORY confirmation gate): Never run `git commit` (or `git commit --amend`) until the exact proposed commit message has been shown to the user in a message and the user has replied with explicit approval. This is a hard stop that overrides any broader instruction.
  - A request to "commit" / "commit and push" / "commit on branch X" authorizes the *action* but NOT the *message*. Always pause and present the message first.
  - Present the message as its own final step — never bundle the message into the same tool call/command that performs the commit (e.g. do not put the message inline in the `git commit -m` call before approval).
  - Approval must be explicit for this commit (e.g. "yes", "go", "lgtm"). Silence, a prior general go-ahead, or approval of the *plan* does NOT count.
  - The ONLY way to skip this gate is an explicit waiver for that commit (e.g. "commit without showing me the message").
  - If you ever commit without approval, treat it as an error: stop, tell the user, show the message actually used, and offer to amend (message-only) or `git reset --soft HEAD~1` per their choice.
  - Keep messages SHORT. Hard limits: subject line <= 88 characters, body <= 6 lines wrapped at 88 characters, and no more than 4 bullets. Shorter is always better; a one-line commit message with no body is a perfectly good outcome and should be the default for focused changes.
  - Write the body only if it adds information the diff does not already convey (why, not what). Do not enumerate every file or function touched — the diff already does that. If a bullet just restates a code change, delete it.
  - Do not put measurements, benchmark tables, verification results, caveats, or design rationale in the commit message. Those belong in `devel/features_request.md` or the relevant `devel/` plan file. Reference the feature id (e.g. `F007:`) and let that file carry the detail.
  - Before presenting a message, count the body lines. If it exceeds 6, cut it down rather than presenting it and asking. Err on the side of terse.
  - If the same prompt also asks to close/wrap up the session, write `devel/devel_ai_log.txt` and stage it BEFORE presenting the commit message. See "Session logging" below; the log belongs in that same commit, not a follow-up one.
- Notebook output clearing (always clear before edit): Before making ANY modification to a Jupyter notebook (`.ipynb`) — including edits that only touch cell `source`, docstrings, comments, or single characters — always clear all cell outputs first. For every code cell set `cell["outputs"] = []` and `cell["execution_count"] = None`. Leave cell `source`, cell IDs, and notebook metadata untouched. Then perform the requested edit in the same write. No prompt is required; this is unconditional. The user re-executes cells after edits as needed. This rule applies regardless of edit size and regardless of which tool (`Edit`, `Write`, etc.) performs the change. This rule is scoped strictly to `.ipynb` files.
- Strip notebook output before committing (never commit outputs): Every Jupyter notebook (`.ipynb`) MUST be fully output-stripped before it is staged or committed to git. This is a hard, non-negotiable requirement: a notebook containing ANY cell output or non-null `execution_count` must NEVER enter the git index. Before running `git add` on (or staging) any `.ipynb`, first verify it is clean; for every code cell `cell["outputs"]` must be `[]` and `cell["execution_count"]` must be `None`. If it is not clean, strip it (in-place, leaving `source`, cell IDs, and notebook metadata untouched) and only then stage it. If a notebook with output is discovered already staged, unstage it, strip it, and re-stage. Do not use `--no-verify` or otherwise bypass this. Prefer verifying the staged content itself (e.g. inspect `git show :<path>` / the staged blob), not just the working-tree copy. This rule is scoped strictly to `.ipynb` files and applies regardless of which tool performs the staging or commit.
- Session logging: When the user asks to close a session, append a concise log of the current session's changes and key findings to `devel/devel_ai_log.txt`. Target length: aim for ~10 lines per session, not 30+. Use terse bullet points; avoid restating function signatures, option dictionaries, or code that already lives in the repo. Required sections: a short header (date, one-sentence goal), "Files modified", "Key changes" (one bullet per change, no nested option lists), "Key findings / gotchas" (only non-obvious ones), and optionally "Suggested follow-ups". Omit any section that has no content rather than padding it.
  - Also copy verbatim any key prompts that were used as part of the log, but avoid clarifying questions or answers that were not material to defining the essential tasks.
  - Single close-out commit (log goes IN the commit): If one prompt asks for both a commit and a session close — in any order or phrasing ("commit the changes and close the session", "close out and commit", "wrap up and commit this") — treat it as ONE close-out operation. Write/append `devel/devel_ai_log.txt` FIRST, stage it together with the rest of the work, and produce a SINGLE commit. Never commit the code and then follow it with a separate log-only commit.
  - The order the tasks are listed in the prompt is NOT an instruction to commit first. "Commit and close" and "close and commit" mean the same thing: log first, then one commit.
  - Because the log is part of the commit, write it before presenting the commit message for approval, so the message being approved describes a diff that already includes the log.
  - Only split the log into its own commit if the code was already committed in an earlier turn, or if separate commits are explicitly requested. If a split seems warranted for any other reason, ask rather than splitting silently.

## Feature Requests

- Creation: Upon identifying a new feature or significant change request, immediately add it to `devel/features_request.md` with the current date and "Pending" status. Put the most recent feature requests at the top of this file.
- Alignment: Before creating a plan, verify alignment with `devel/features_request.md`. If the task is new, create a tracking entry first.
- Completion: Upon successful implementation and verification, ask the user if the feature is done. If so mark as "Done" in `devel/features_request.md` and include a brief note on the implementation details.
- Summary tables: `devel/features_request.md` opens with a "Pending" and a "Done" summary table (feature id, one-phrase description, start date, status), each sorted highest feature number first. Whenever a feature entry is added, moved between Pending/Done, or has its status/description changed, update the matching row in both tables in the same edit.
- Plan files: store feature implementation plans under `devel/plans/`, named `plan_F0XX_<slug>.md` using the tracked feature's id (the lowest id if a plan spans a range, e.g. `plan_F014_lalston_integration.md` for F014-F019). Reference the plan file from its feature entry/entries in `devel/features_request.md`.

## Coding Standards

- Use Python 3.10+ features (e.g., modern type hinting with `list[...]` not `List[...]`).
- Maintain JSON compatibility for all input configuration containers.
- Maintain hdf5/h5py compatibility for all output dictionaries.
- Visualization tools (Plotly) should handle metadata keys (like `runid`) gracefully.
- Prefer spaces to tabs in all situations.
- Follow NumPy-style docstrings (project uses `flake8-docstrings` with numpy convention).
- Use `ii`, `jj`, `kk` instead of `i`, `j`, `k` for loop variable names (avoids shadowing and improves searchability).
- Avoid the use of embedded or nested functions. Instead make these private module or class level functions. Closures/local body functions for jax lax/vmap transforms are permitted.
- Format code with Black (line length 88) and sort imports with isort (profile: black).
