.PHONY: reproduce-smoke reproduce benchmark test research-invariants lint format-check typecheck rust-test rust-equivalence

UV_RUN = UV_CACHE_DIR=/tmp/bist-uv-cache PYTHONPATH=src uv run
RUNS_ROOT ?= runs

reproduce-smoke:
	$(UV_RUN) bist-predict reproduce-smoke --runs-root $(RUNS_ROOT)

reproduce:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" && exit 2)
	$(UV_RUN) bist-predict reproduce $(RUN_ID) --runs-root $(RUNS_ROOT)

benchmark:
	@test -n "$(INPUT)" || (echo "INPUT is required" && exit 2)
	$(UV_RUN) bist-predict benchmark --prices $(INPUT) --runs-root $(RUNS_ROOT)

test:
	$(UV_RUN) pytest -q

research-invariants:
	$(UV_RUN) pytest tests/test_research tests/test_ingest/test_calendar.py tests/test_ingest/test_corporate_actions.py tests/test_ingest/test_reconciliation.py -q

lint:
	$(UV_RUN) ruff check src tests

format-check:
	$(UV_RUN) ruff format --check src tests

typecheck:
	$(UV_RUN) mypy src

rust-test:
	cargo test --workspace

rust-equivalence:
	$(UV_RUN) pytest tests/test_features/test_rust_equivalence.py -q
