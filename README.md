# CcrBench

CcrBench tells whether a model solves a problem by **reasoning** or by **memorizing** it.
It solves each problem many times in isolated sessions (Cross-Context Review), measures how
much the solution paths differ, and checks how accuracy changes when the problem is
rewritten on the surface. It needs only API access to the model, no weights, and fits in CI:

```
ccrbench run suite.yaml --model anthropic:<model>   # diversity + variant robustness, HTML/JSON report
```

Benchmark contamination is a credibility problem. Text-overlap and perplexity checks look at
what a model *outputs*; CcrBench looks at *how it gets there*, and reports two independent
signals instead of one number.

> **A label is a sign, not a verdict.** "Suspected memorization" (`suspected`) means the
> evidence points that way under the thresholds you set. It is probabilistic by design: easy
> problems with a single natural solution can look uniform without any contamination, which
> is why a second, independent signal is required. Reports ship the raw transcripts' paths so
> every score can be reproduced and challenged.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Suite file](#suite-file)
- [The two axes and the label](#the-two-axes-and-the-label)
- [Variants](#variants)
- [Cost control](#cost-control)
- [Isolation and the run store](#isolation-and-the-run-store)
- [Providers](#providers)
- [Reports and CI](#reports-and-ci)
- [Security](#security)
- [Python API](#python-api)
- [Development](#development)
- [License](#license)

## Install

CcrBench is managed with [`uv`](https://docs.astral.sh/uv/). It is not on PyPI yet; install
from a clone:

```
git clone <this repository>
cd CcrBench
uv sync --locked --extra all        # everything: the three provider SDKs
uv run ccrbench --version
```

Extras pull in the provider SDKs you need (the OpenAI-compatible adapter for local models
needs none of them):

| Extra | Installs | Adapter prefix |
| --- | --- | --- |
| `anthropic` | `anthropic` | `anthropic:` |
| `openai` | `openai` | `openai:` |
| `gemini` | `google-genai` | `gemini:` |
| `all` | all of the above | |

e.g. `uv sync --locked --extra anthropic`. Without the extra, the adapter fails with a message
that names it.

### `tunedness-embed`: a git dependency

Transcript embeddings are computed locally with an ONNX model (no API, no cost, transcripts
never leave your machine). That code lives in a separate library, **TunedEmbed**, which is
not on PyPI. `pyproject.toml` pins it to a tag of its public repository:

```toml
[tool.uv.sources]
tunedness-embed = { git = "https://github.com/Tunedness/TunedEmbed", tag = "v0.1.0" }
```

To develop against a local checkout of TunedEmbed instead, point the source at the path
(do not commit this change):

```toml
[tool.uv.sources]
tunedness-embed = { path = "../TunedEmbed", editable = true }
```

then run `uv lock && uv sync --extra all`. For a one-off without editing anything, use
`uv run --with-editable ../TunedEmbed ccrbench ...`.

### Python version

`.python-version` pins **3.14**. Do not bump it to 3.15 yet: `onnxruntime`, which TunedEmbed
needs, publishes no `cp315` wheel. CcrBench itself supports Python 3.11 to 3.14; to use
another interpreter for one command, set `UV_PYTHON`, e.g. `UV_PYTHON=3.12 uv sync --locked --extra all`.

## Quick start

```
uv run ccrbench validate examples/suite.yaml                  # schema check, prints the suite hash
uv run ccrbench estimate examples/suite.yaml --model anthropic:<model>
uv run ccrbench run examples/suite.yaml --model anthropic:<model>
uv run ccrbench report runs/ --suite examples/suite.yaml      # rebuild the report from stored runs
```

`--model` is always `adapter:model`. `run` prints the cost estimate, runs the matrix with one
progress line per session (on stderr), writes the reports and prints a summary table, for
example:

```
problem        diversity  accuracy  variant drop  label
reverse-words  0.21       1.00      0.85          suspected
train-distance 0.64       1.00      0.00          not_suspected
```

Files are written to `reports/<suite-hash>/report-<adapter>__<model>.json` and `.html`
(`--out DIR` changes the directory, `--format json` or `--format html` selects one).
Transcripts go to `runs/` (`--runs-dir`). Both directories are git-ignored in this repository.

Commands:

| Command | What it does |
| --- | --- |
| `ccrbench validate SUITE` | Validates the suite, prints the problem count and hash. |
| `ccrbench estimate SUITE --model M` | Cost estimate of a run, originals plus variants; runs nothing. |
| `ccrbench run SUITE --model M` | The full pipeline: variants, budget gate, runs, metrics, scoring, report. |
| `ccrbench report RUNS_DIR --suite SUITE` | Rebuilds the report from the store without calling any model. `--model` is needed when the store holds several. |

`run` options: `--runs-dir`, `--out`, `--format`, `--sample K`, `--yes`, `--pricing FILE`,
`--unsafe-local-exec`, `--allow-unknown-cost`, `--docker-image`.

## Suite file

A suite is a YAML file; unknown fields are rejected. A complete example is
[`examples/suite.yaml`](examples/suite.yaml).

| Field | Meaning |
| --- | --- |
| `name`, `version` | Identify the suite (both are part of its hash). |
| `problems` | The problems (below). Ids are slugs: `[a-z0-9][a-z0-9_-]*`. |
| `runs` | Isolated sessions per problem and per variant. Default `10`. |
| `variation` | `temperatures` (default `[0.7]`) and `prompt_templates` (default `["{prompt}"]`; each must contain `{prompt}`). Runs cycle through the combinations. |
| `budget_usd` | Cost ceiling; see [Cost control](#cost-control). |
| `max_suspected` | CI tolerance: fail only when *more* than this many problems are `suspected`. Unset: any one fails. |
| `thresholds` | `low_diversity` (0.3), `high_accuracy` (0.8), `high_variant_drop` (0.3). Provisional defaults, to be calibrated. |
| `judge` | `model: adapter:model`, required when a problem uses `judge_template`. |
| `variant_transforms` | Automatic variant transforms; default `[rename_identifiers, substitute_numbers]`. |
| `description` | Free text, not part of the hash. |

A problem has an `id`, a `kind` (`code`, `question` or `proof`), a `prompt`, an optional
`language`, optional hand-written `variants`, and a `check`, how a response counts as correct:

| `check.type` | Correct when | Fields |
| --- | --- | --- |
| `expected_answer` | the response's final answer equals `answer` after `normalize` | `answer`, `normalize` (`exact`, `casefold_strip`, `numeric`), `answer_expr` |
| `test_command` | `command` exits 0 in a directory holding `files` plus the model's code | `command` (list), `files`, `timeout_s` |
| `judge_template` | the judge model's verdict equals `pass_label` | `template` (with `{prompt}`/`{response}`), `pass_label` |

The final answer is the last `Answer:` line of the response, else its last non-empty line.
For code, the last fenced block is written to `solution.<ext>` next to `files`.
`answer_expr` is an arithmetic expression over the prompt's integers (`n0`, `n1`, ...) that
lets the number transform recompute the expected answer.

### YAML restrictions

Suites are parsed by a restricted loader. **Anchors and aliases (`&name`, `*name`) and the
merge key (`<<:`) are not supported**: an alias lets a tiny file expand into a huge
structure, so they are rejected with an error that names the line. Duplicate keys are
rejected too, the file must be UTF-8 and at most 4 MiB. Spell repeated blocks out.

## The two axes and the label

CcrBench never collapses its result into a single number (ADR-004).

1. **Path diversity** (0 to 1, high means diverse), a confidence-weighted mean of four
   metrics: step-sequence edit distance, embedding dispersion of the transcripts, AST
   difference (code only) and branching entropy at decision points. Steps are split by a
   deterministic heuristic (numbered steps, paragraphs, code blocks); no output format is
   imposed on the model. The embedding model's id is recorded in the report.
2. **Variant robustness**: accuracy on the original problem versus on its surface-changed
   variants, and the drop between them.

A problem is labelled **`suspected`** only when all three hold: *low diversity* (score below
`low_diversity`), *high original accuracy* (at least `high_accuracy`) and *high variant drop*
(at least `high_variant_drop`). If an axis cannot be computed (no variant could be derived,
too few runs) the label is `undetermined` with the reason, never a guess. Everything else is
`not_suspected`. When every problem is `undetermined` the exit status says nothing about
memorization; the summary says so.

## Variants

Variants are generated deterministically, so they are reproducible and part of the run
identity. Built-in transforms:

- `rename_identifiers`: renames Python identifiers consistently in the prompt's code and in
  the `.py` test files; skipped when that cannot be proven safe.
- `substitute_numbers`: replaces the prompt's integers and recomputes the answer with
  `answer_expr`; skipped for problems without one.
- `permute_order`: **opt-in.** It reorders lists in the prompt, but cannot tell from text
  whether the answer depends on the order ("start with", "number 2", ...), so it is off
  unless the suite lists it in `variant_transforms`. Enable it only for problems where the
  order is irrelevant.

A transform that cannot derive a variant leaves a skipped entry with its reason in the
report. Hand-written `variants` on a problem (each with its own `prompt` and `check`) run
alongside the automatic ones. More transforms and metrics can be added as plugins through the
`ccrbench.variants` and `ccrbench.metrics` entry point groups.

## Cost control

A run is `runs` sessions per problem and per variant (default 10), so it is not free.

- **Preview.** `run` and `estimate` show total, stored and billable runs, token counts and a
  cost *ceiling* (output priced at `max_tokens` for every run) before anything is sent.
- **`budget_usd`.** If the estimate exceeds it, an interactive session asks for
  confirmation (`--yes` answers it); **without a terminal (CI) the run does not start**
  (exit code 2).
- **`--sample K`** runs a seeded subset of K problems.
- **Stored runs cost nothing.** A repeated run reads transcripts from the store and makes no
  model call. Only the runs of the same model count as paid.

### Prices: `pricing.yaml`

The bundled price table (`src/ccrbench/data/pricing.yaml`) lists standard-tier prices for
current Anthropic, OpenAI and Gemini models, each copied from the provider's official pricing
page with that page in `source` and the check date in `as_of`. Where a provider tiers by prompt
length, the lowest tier is listed and the higher tier is noted in a comment. Prices change:
treat the table as a snapshot and check `source` before relying on a cost preview.

A model that is not listed has an unknown price: the cost preview says "price unknown", and a
suite with `budget_usd` refuses to start because it cannot check the budget. Provide your own
table with `--pricing` (it replaces the bundled one):

```yaml
models:
  your-model-name:                  # the part after "adapter:"
    input_usd_per_mtok: 1.00        # USD per 1M tokens, from the provider's pricing page
    output_usd_per_mtok: 5.00
    source: https://provider.example/pricing
    as_of: 2026-11-14
```

`--allow-unknown-cost` runs a budgeted suite anyway when a price is unknown. Local models
have no price on purpose.

## Isolation and the run store

Every run is an independent call sequence: no message is carried between sessions. Each
system message carries a session nonce derived from `(suite hash, problem, run index)`: it
differs between runs, which defeats provider-side caching, and is identical when a run is
repeated, which keeps the local store valid. Provider cache parameters are switched off where
the provider allows it; the report warns where it cannot be.

Transcripts are stored content-addressed:

```
runs/<suite-hash>/<adapter>__<model>/<problem>/run-<N>.json
```

Models never see each other's records, and the report cites the run files behind every score.
Records made with different inputs (another `max_tokens` or prompt template) are refused
instead of overwritten; use a fresh `--runs-dir`.

## Providers

| Prefix | Credentials | Notes |
| --- | --- | --- |
| `anthropic:` | `ANTHROPIC_API_KEY` | see below |
| `openai:` | `OPENAI_API_KEY` | |
| `gemini:` | `GEMINI_API_KEY` | |
| `openai-compat:` | `OPENAI_COMPAT_BASE_URL`, optional `OPENAI_COMPAT_API_KEY` | Ollama, vLLM, any OpenAI-compatible endpoint; model names may contain colons (`openai-compat:llama3:8b`) |
| `fake:` | none | `fake:memorize` and `fake:diverse`: deterministic stand-ins for tests and demos |

### Anthropic: `temperature` and `seed` are ignored

The current Anthropic Messages API (version `2026-10-21`) rejects `temperature` and has no
`seed`, so the adapter does not send them and declares them ignored. Consequently the
temperature and seed variation axes are **not applied** to Claude models: diversity comes
only from the prompt-template variation and the model's own sampling. **The report says so**:
every affected problem carries an "ignored parameter" warning ("this axis was not applied"),
and the same warning appears for any other provider that drops a parameter. Read diversity
scores of such models with that in mind; adding `prompt_templates` to `variation` is the way
to keep a variation axis alive.

## Reports and CI

The JSON report (schema in `src/ccrbench/report/schema/report.schema.json`) and the
single-file HTML report hold, per problem: both axes with their per-metric breakdown, the
label with its three conditions, skipped variants with reasons, and the run files that back
the scores; plus the thresholds used (default or from the suite), the embedding model id and
all warnings.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | no problem is `suspected` (or no more than `max_suspected`) |
| `1` | at least one `suspected` problem (more than `max_suspected` when set) |
| `2` | configuration, budget or runtime error: invalid suite, closed budget gate, missing Docker or credentials, provider failure, unwritable `--runs-dir` or `--out`, aborted prompt |
| `3` | internal error: an unexpected failure (a bug in CcrBench); it is reported on stderr and is never mistaken for a `suspected` result |

Only `1` means "suspected", so a CI job can tell a finding from a failure.

A failed provider call stops the run with exit code 2; transcripts already paid for are kept,
so re-running resumes where it stopped.

[`examples/github-actions.yml`](examples/github-actions.yml) is a workflow that installs `uv`,
runs `ccrbench run`, uploads the report as an artifact and fails the job through the exit code.

## Security

`test_command` checks execute **model-generated code**.

- By default they run in a throwaway **Docker** container: no network, read-only root
  filesystem, memory/CPU/process limits, a non-root user, and no provider credentials. The
  image (`python:3.12-slim` by default, `--docker-image` to change) is never pulled
  implicitly; `docker pull` it first. It must contain the tools your `command` needs: the
  default image has no `pytest`, so the example suite uses `unittest`.
- **`--unsafe-local-exec` runs the code directly on your machine, with your user's
  permissions and no isolation.** The code can read your files and use your network. Use it
  only in a disposable environment you already trust to be destroyed, and never on a machine
  with credentials you care about. It prints a warning on every execution.
- Without Docker and without `--unsafe-local-exec`, a suite containing `test_command`
  checks does not run (exit code 2).
- A suite file chooses the commands that run in the container: treat third-party suites like
  code and read them before running.

Provider keys are read from the environment; the code under test never sees them (the
container gets no environment from CcrBench, and the local executor passes a minimal one).

## Python API

`ccrbench.api` is a thin layer over the functions the CLI calls:

```python
from pathlib import Path

from ccrbench.api import RunOptions, build_report, load_suite, run_suite

outcome = run_suite("suite.yaml", "anthropic:<model>", RunOptions(runs_dir=Path("runs")))
print(outcome.report.summary.suspected, outcome.exit_code)

report = build_report("suite.yaml", "anthropic:<model>", "runs")  # from the store, no model calls
suite = load_suite("suite.yaml")
```

`run_suite` raises the loader, adapter and runner errors that the CLI turns into exit code 2.
A run that would exceed `budget_usd` only starts when `RunOptions(interactive=True)` lets the
runner ask (`yes=True` then answers for you, `confirm=` replaces the prompt); otherwise it raises
`BudgetGateError`, exactly like the CLI without a terminal.

## Development

```
uv sync --locked --extra all
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

The default `pytest` run needs no network and no Docker; tests marked `network`, `live` and
`docker` are deselected. End-to-end tests run the whole pipeline with the fake adapters and an
offline hash embedder.

## License

[Apache-2.0](LICENSE).
