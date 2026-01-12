# Coevolved Code Contribution Guidelines

First things first, thanks for even being interested in contributing!

All types of contributions are encouraged and valued. See the [Table of
Contents](#table-of-contents) for different ways to help and details about
how this project handles them. Please make sure to read the relevant section
before making your contribution. It will make it a lot easier for us
maintainers and smooth out the experience for all involved. The community
looks forward to your contributions.

> And if you like the project, but just don't have time to contribute, that's
> okay! There are other easy ways to support the project and show your
> appreciation, which we would also be very happy about:
>
> - Star the project
> - Tweet about it
> - Reference this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Local Development Setup](#local-development-setup)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
- [Acceptable LLM Usage](#acceptable-llm-usage)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of
Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold
this code. Please report unacceptable behavior to conduct@coevolved.ai.

## I Have a Question

> If you want to ask a question, we assume that you have read the available
> [Documentation](https://docs.coevolved.ai).

Before you ask a question, it is best to search for existing
[Issues](https://github.com/coevolved/coevolved/issues) that might help you. In
case you have found a suitable issue and still need clarification, you can write
your question in this issue. It is also advisable to search the internet for 
answers first.

If you then still feel the need to ask a question and need clarification, we
recommend the following:

- Open an [Issue](https://github.com/coevolved/coevolved/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions, depending on
  what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice
>
> When contributing to this project, you must agree that you have authored
> this content, that you have the necessary rights to the content and
> that the content you contribute may be provided under the project license.
> Usage of AI coding tools can be found in our
> [statement on LLM usage](#acceptable-llm-usage).

### Local Development Setup

This project uses [uv](https://astral.sh/uv) for fast and reliable Python
package management.

#### Prerequisites

- Python 3.9 or higher
- [uv](https://astral.sh/uv) package manager

Install uv:

```bash
pip install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Setup Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/coevolved/coevolved.git
   cd coevolved
   ```

2. **Create virtual environment and install dependencies**:

   ```bash
   uv venv
   source .venv/bin/activate  # On macOS/Linux
   # or .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:

   ```bash
   # Install runtime dependencies only (what end users get)
   uv sync --no-group

   # Install with dev dependencies (recommended for contributors)
   uv sync

   # Install with cookbook dependencies (for running examples)
   uv sync --group cookbook
   ```

   Note: `uv sync` automatically installs the `coevolved` package in editable 
   mode, so code changes take effect immediately without reinstalling.

#### Understanding Dependencies

**Runtime dependencies** (`[project].dependencies`):

- Required for the `coevolved` package to function
- Automatically installed when users run `pip install coevolved`
- Examples: `pydantic`, `uuid_utils`

**Optional dependencies** (`[project].optional-dependencies`):

- **`dev`**: Development tools (linting, formatting, testing)
  - Installed with `uv sync` (default for local development)
  - Examples: `ruff`
- **`cookbook`**: Dependencies needed only for running examples
  - Installed with `uv sync --group cookbook`
  - Examples: `jupyter`, `matplotlib`, `pandas`
  - **Not included** in the published PyPI package

#### Adding New Dependencies

When adding dependencies, choose the appropriate location:

- **Runtime dependencies**: Add to `[project].dependencies` if required for the
    package to work
- **Dev dependencies**: Add to `[project].optional-dependencies.dev` for
    development tools
- **Cookbook dependencies**: Add to `[project].optional-dependencies.cookbook`
    for example scripts only

After modifying `pyproject.toml`:

```bash
uv sync
git add pyproject.toml uv.lock
```

### Reporting Bugs

Use the latest release and search existing issues. Then open a new
[issue](https://github.com/coevolved/coevolved/issues/new) with a clear title
and the following:

- Expected vs. actual behavior
- Reproduction steps (ideally a minimal test case)
- Environment details (OS, runtime versions, stack trace or logs)
- Any extra context that helps us verify quickly

> <span style="display: block; background-color: rgba(220, 38, 38, 0.15)
font-weight: bold; padding: 0.75em 1em; border-radius: 4px;">
> Security or sensitive reports must go to <a href="mailto:security@coevolved.ai">security@coevolved.ai</a>; never the public issue tracker.
> </span>

### Suggesting Enhancements

For feature ideas or design improvements, search existing issues. Then open
a [suggestion](https://github.com/coevolved/coevolved/issues/new) that covers:

- The problem or use case
- Your proposed approach and any alternatives considered
- Context, screenshots, or examples showing the user benefit

## Acceptable LLM Usage

Generative AI can be a great accelerant for solving problems or exploring
solutions if used thoughtfully. We encourage contributors to leverage these
tools for brainstorming, code suggestions, or kickstarting documentation, but
you are ultimately responsible for what you submit.

**Please do not open pull requests that are entirely LLM-generated without
careful human review.** We believe that such submissions often lack context, 
accuracy, and alignment with project standards. Unreviewed AI content (code, 
docs, or issue text) will be closed or declined without regard.

If you use LLMs:

- Review and test all generated code as if you wrote it from scratch.
- Adapt and contextualize suggestions for this project—don’t just paste and 
    submit.
- Own the results, including fixing errors or following up on reviewer feedback.

Thoughtful, well-checked contributions are always welcome regardless of your 
workflow!

## Styleguides

- Code style: follow PEP 8; format with black; sort imports with isort.
  Type all new Python code and keep it mypy-friendly.
- Dependencies: prefer the standard library; justify new packages in the PR.
- Docs/comments: keep docstrings current; add brief comments only when
  intent is not obvious; update README or usage examples when behavior
  changes.
- Performance/security: call out perf-sensitive paths and any security
  impact in the PR; never log secrets; use environment variables for
  credentials.
- PR etiquette: keep PRs small and focused, use clear titles (Conventional
  Commits), and include a short summary of what/why plus test results.

### Commit Messages

Please use descriptive and comprehensive commit messages and pull request
titles when contributing. As a whole, Coevolved uses Conventional Commit
messages based on the Angular convention.

If you are new to Conventional Commits and want to make the process easier,
consider using [CommitWiz](https://www.npmjs.com/package/commitwiz), an
out-of-the-box CLI tool for git staging and writing commits.
