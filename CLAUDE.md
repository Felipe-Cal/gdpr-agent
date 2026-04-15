# Claude Code — Project Instructions

## Git workflow

**Always use feature branches. Never commit directly to `master`.**

For every task:
1. Create a branch: `git checkout -b <type>/<short-description>`
   - Use conventional prefixes: `feat/`, `fix/`, `docs/`, `chore/`
   - Example: `feat/phase4-vllm-serving`, `fix/groundedness-metric`, `docs/phase4`
2. Implement the changes with one or more commits.
   - **Never add `Co-Authored-By:` or any Claude attribution to commit messages or PR bodies.**
3. Push the branch: `git push -u origin <branch>`
4. Open a PR with `gh pr create` — include a summary and test plan. No attribution footer.
5. **Stop. Do not merge.** The user reviews and merges the PR.

## Cost discipline

This project runs on a real GCP billing account with no free credits.

Before running any command that incurs cost (cluster creation, GPU node pools,
Vertex AI training jobs, Cloud Run deployments, etc.):
- Say explicitly: **"⚠️ This will start billing. Estimated cost: $X/hr."**
- Wait for the user to confirm before proceeding.

After any billable resource is created, always remind the user how to tear it down:
- Include the exact `gcloud` / `kubectl` delete command.
- Repeat the reminder when the session ends or when switching tasks.

## Code style

- Python 3.11+, type hints throughout.
- Pydantic-settings for all configuration — no hardcoded values.
- `europe-west4` for all GCP resources (EU data sovereignty).
- Rich for CLI output. Typer for CLI argument parsing.
- Comments explain *why*, not *what*.

## Project context

This is a GDPR Legal Analyst Agent built as a structured learning project
covering the GCP AI/ML stack for a senior ML architect role at Zencore.
Each phase teaches a specific cluster of frameworks — keep that framing when
writing docs and comments. See `docs/phases-overview.md` for the full roadmap.
