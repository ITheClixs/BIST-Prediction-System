.PHONY: reproduce-smoke reproduce benchmark test research-invariants lint format-check typecheck coverage rust-test rust-equivalence provider-smoke

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
	$(UV_RUN) pytest tests/test_research --ignore=tests/test_research/test_rust_benchmark.py tests/test_ingest/test_calendar.py tests/test_ingest/test_corporate_actions.py tests/test_ingest/test_reconciliation.py -q

lint:
	$(UV_RUN) ruff check src tests

format-check:
	$(UV_RUN) ruff format --check src tests

typecheck:
	$(UV_RUN) mypy src

coverage:
	$(UV_RUN) coverage run -m pytest -q
	$(UV_RUN) coverage report

rust-test:
	cargo test --workspace

rust-equivalence:
	$(UV_RUN) pytest tests/test_features/test_rust_equivalence.py -q

provider-smoke:
	$(UV_RUN) python -m bist_predict.ingest.provider_smoke --ticker THYAO --days 21
