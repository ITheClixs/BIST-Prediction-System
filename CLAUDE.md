## Workflow Orchestration\
\
### 1. Plan Mode Default\
•⁠  ⁠Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)\
•⁠  ⁠If something goes sideways, STOP and re-plan immediately\
•⁠  ⁠Use plan mode for verification steps, not just building\
•⁠  ⁠Write detailed specs upfront to reduce ambiguity\
\
### 2. Subagent Strategy\
•⁠  ⁠Use subagents liberally to keep main context window clean\
•⁠  ⁠Offload research, exploration, and parallel analysis to subagents\
•⁠  ⁠For complex problems, throw more compute at it via subagents\
•⁠  ⁠One task per subagent for focused execution\
\
### 3. Self-Improvement Loop\
•⁠  ⁠After ANY correction from the user: update tasks/lessons.md with the pattern\
•⁠  ⁠Write rules for yourself that prevent the same mistake\
•⁠  ⁠Ruthlessly iterate on these lessons until mistake rate drops\
•⁠  ⁠Review lessons at session start for relevant project\
\
### 4. Verification Before Done\
•⁠  ⁠Never mark a task complete without proving it works\
•⁠  ⁠Diff behavior between main and your changes when relevant\
•⁠  ⁠Ask yourself: "Would a staff engineer approve this?"\
•⁠  ⁠Run tests, check logs, demonstrate correctness\
\
### 5. Demand Elegance (Balanced)\
•⁠  ⁠For non-trivial changes: pause and ask "is there a more elegant way?"\
•⁠  ⁠If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"\
•⁠  ⁠Skip this for simple, obvious fixes -- don't over-engineer\
•⁠  ⁠Challenge your own work before presenting it\
\
### 6. Autonomous Bug Fixing\
•⁠  ⁠When given a bug report: just fix it. Don't ask for hand-holding\
•⁠  ⁠Point at logs, errors, failing tests -- then resolve them\
•⁠  ⁠Zero context switching required from the user\
•⁠  ⁠Go fix failing CI tests without being told how\
\
## Task Management\
\
1.⁠ ⁠Plan First: Write plan to tasks/todo.md with checkable items\
2.⁠ ⁠Verify Plan: Check in before starting implementation\
3.⁠ ⁠Track Progress: Mark items complete as you go\
4.⁠ ⁠Explain Changes: High-level summary at each step\
5.⁠ ⁠Document Results: Add review section to tasks/todo.md\
6.⁠ ⁠Capture Lessons: Update tasks/lessons.md after corrections\
\
## Core Principles\
\
•⁠  ⁠Simplicity First: Make every change as simple as possible. Impact minimal code.\
•⁠  ⁠No Laziness: Find root causes. No temporary fixes. Senior developer standards.\
•⁠  ⁠Minimal Impact: Only touch what's necessary. No side effects with new bugs.\

## Project: BIST Predictor

### Quick Start
```bash
uv sync                        # Install dependencies
uv run bist-predict --help     # CLI usage
uv run pytest tests/ -v        # Run all tests
```

### Architecture
Modular pipeline: Data Ingest → Feature Engine (Rust) → Quant Alpha → ML Ensemble → Evaluation
- Design spec: `docs/superpowers/specs/2026-04-02-bist-predictor-design.md`
- Implementation plans: `docs/superpowers/plans/`

### Key Conventions
- All data types in `src/bist_predict/ingest/types.py` — use these, don't create ad-hoc dicts
- Database access via `Database` class only — never raw sqlite3 outside storage/
- Collectors implement `PriceCollector`, `MacroCollector`, or `SentimentCollector` protocols
- All async I/O — use `httpx.AsyncClient`, wrap sync libs with `asyncio.to_thread`
- Tests use `respx` for HTTP mocking, `tmp_db_path` fixture for database tests
- Config loaded via `load_config()` — never hardcode paths or API keys

### Testing
```bash
uv run pytest tests/ -v                        # All tests
uv run pytest tests/test_ingest/ -v             # Ingest tests only
uv run pytest tests/test_storage/ -v            # Storage tests only
uv run pytest tests/path/test_file.py::test_fn  # Single test
```