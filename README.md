<p align="center">
  <picture class="github-only">
    <source media="(prefers-color-scheme: light)" srcset="https://coevolved.github.io/coevolved-assets/static/wordmark_light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="https://coevolved.github.io/coevolved-assets/static/wordmark_dark.svg">
    <img alt="Coevolved Logo" src="https://coevolved.github.io/coevolved-assets/static/wordmark_light.svg" width="80%">
  </picture>
</p>

<div>
<br>
</div>

[![PyPI](https://img.shields.io/pypi/v/coevolved.svg)](https://pypi.org/project/coevolved/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Y Combinator W26](https://img.shields.io/badge/Y%20Combinator-W26-orange)](https://www.ycombinator.com/companies/coevolved)


The Coevolved Agent Development Framework is a flexible, atomic-first framework
that gives software developers more control over AI agent construction. It is an
efficient orchestration framework for building, debugging, managing, and
deploying agent systems in production.

## Installation

Install the `coevolved` package from PyPI:

```bash
pip install -U coevolved
```

## Value Adds

- **Atomic-first Architecture & Observability**: Every operation is a `Step`, a
  composable unit with built-in input/output validation, automatic tracing, and
  execution metadata. Steps emit structured events (start, end, error) with
  configurable snapshot policies, giving you complete visibility into agent
  execution without manual instrumentation.

- **Checkpointing**: Built-in state persistence infrastructure enables failure
  recovery, human-in-the-loop workflows, and time-travel debugging. Resume from
  any checkpoint with full state lineage, making long-running agents reliable
  and debuggable.

- **Composable Patterns**: Express complex workflows using built-in composition
  primitives—sequential chains, parallel execution, fallback strategies, and
  retry logic with exponential backoff. Compose steps into agents without
  writing custom orchestration code.

- **Budget Enforcement**: Enforce limits on steps, LLM calls, tokens, cost, and
  execution time with `UsagePolicy` and `UsageTracker`. Get automatic warnings
  at configurable thresholds and prevent runaway costs in production
  deployments.

- **Type-safe Validation**: Optional Pydantic schemas provide runtime type
  validation where you need it, without forcing rigid type systems. Steps
  validate inputs and outputs automatically, catching errors early while
  remaining flexible.

## Contributing

We welcome contributions from the community! Whether it's a feature request, bug
report, documentation improvement, or code contribution, please read our

- [Code Contribution Guidelines](./CONTRIBUTING.md)

## Additional Resources

More documentation can be found at our
[docs](https://docs.coevolved.ai).

### Local Development

For developers who want to contribute or run Coevolved locally, we support a
fast and reliable workflow powered by [`uv`](https://astral.sh/uv), an ultrafast
Python package manager and virtual environment tool.

When you run `uv sync`, it automatically installs the `coevolved` package in
editable mode, so your code changes are immediately available without
reinstalling.

> Detailed local development setup instructions, including dependencies, virtual
> environment creation, and workflow tips, are available
> [here](./CONTRIBUTING.md#local-development-setup).
