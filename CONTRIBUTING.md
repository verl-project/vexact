# Contributing to VeXact

Thank you for your interest in contributing to VeXact! We welcome bug fixes, new features, documentation improvements, and feedback.

Ways to contribute:

- Report bugs or unexpected behaviors (e.g., non-deterministic outputs, memory issues).
- Suggest or implement new features (new model support, attention backends, profiling tools).
- Improve or expand documentation.
- Review pull requests and help other contributors.

## Contributor License Agreement (CLA)

**Before your PR can be merged, you must sign the Contributor License Agreement (CLA).** This is a one-time requirement. The CLA bot will automatically comment on your pull request with signing instructions — please complete it promptly.

## Development Environment

Set up the environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra gpu --extra dev
source .venv/bin/activate
```

- `--extra gpu` — installs PyTorch, FlashAttention 2/3/4
- `--extra dev` — installs `pytest` and `pre-commit` etc.

## Code Linting and Formatting

We use [pre-commit](https://pre-commit.com/) to enforce formatting and linting. Install and run it before submitting a PR:

```bash
uv pip install pre-commit
pre-commit install       # installs git hooks
pre-commit run --all-files  # run all hooks manually
```

Hooks include:

- **ruff** — Python linting (`--fix`) and formatting
- **mdformat** — Markdown formatting (GFM + mkdocs)
- **uv-lock** — ensures `uv.lock` is in sync with `pyproject.toml`
- **pre-commit-hooks** — AST validity, large files, merge conflicts, debug statements, trailing whitespace, end-of-file newline
- **check-license** — verifies license headers in `vexact/`, `tests/`, and `scripts/`

## Testing

```bash
pytest -s tests
```

If your change touches attention, KV cache, sampling, or scheduling, also run the batch invariance tests:

```bash
export model_dir=/path/to/model
. scripts/run_batch_invariant_tests.sh
```

Please add tests for new features or bug fixes. If tests are not applicable, explain why in the PR description.

## Pull Requests

- **Title format**: e.g., `feat: add FlexAttention pipeline parallel support`.
- **Keep PRs focused**: one logical change per PR.
- **Sign the CLA**: required before merge (see above).
- Ensure all unit tests and lint checks pass.

## Architecture Notes for Contributors

Key invariants to preserve:

- **Batch invariance**: Outputs must be identical regardless of how requests are batched — the core correctness guarantee. Do not introduce operations that break determinism across batch configurations.
- **Attention implementations**: Valid `attn_impl` values for `ModelConfig` are `"fa-invariant"` and `"flex"`. Do not use `"flash_attention_3"` in `ModelConfig` (only valid for verification scripts).
- **Configuration centralization**: All configuration defaults live in `config.py` as frozen dataclasses. Do not scatter defaults across the codebase.
- **Comments in English**: All code comments must be in English.

## Thank You

Your contributions help make VeXact a more reliable and capable tool. Happy coding!
