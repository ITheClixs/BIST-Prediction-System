.PHONY: doc-checks reproduce-smoke reproduce reproduce-committed benchmark readme-results test research-invariants lint format-check typecheck coverage rust-test rust-equivalence provider-smoke mutation-check verify-claims figures

UV_RUN = UV_CACHE_DIR=/tmp/bist-uv-cache PYTHONPATH=src uv run
RUNS_ROOT ?= runs
COMMITTED_RUN_ID ?= 20260728T223101Z-8b27df3-2a71b8

reproduce-smoke:
	$(UV_RUN) bist-predict reproduce-smoke --runs-root $(RUNS_ROOT)

reproduce:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" && exit 2)
	$(UV_RUN) bist-predict reproduce $(RUN_ID) --runs-root $(RUNS_ROOT)

reproduce-committed:
	$(MAKE) reproduce RUN_ID=$(COMMITTED_RUN_ID) RUNS_ROOT=$(RUNS_ROOT)

benchmark:
	@test -n "$(INPUT)" || (echo "INPUT is required" && exit 2)
	@test -n "$(ACTIONS)" || (echo "ACTIONS is required" && exit 2)
	@test -n "$(ACTION_COVERAGE)" || (echo "ACTION_COVERAGE is required" && exit 2)
	$(UV_RUN) bist-predict benchmark --prices $(INPUT) --corporate-actions $(ACTIONS) --corporate-action-coverage $(ACTION_COVERAGE) --runs-root $(RUNS_ROOT)

readme-results:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" && exit 2)
	$(UV_RUN) python -m bist_predict.research.readme_results --readme README.md --run $(RUNS_ROOT)/$(RUN_ID)

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

figures:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" && exit 2)
	$(UV_RUN) python tools/build_figures.py --run $(RUNS_ROOT)/$(RUN_ID)

verify-claims:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" && exit 2)
	$(UV_RUN) python tools/verify_claims.py --run $(RUNS_ROOT)/$(RUN_ID)

doc-checks:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" && exit 2)
	$(UV_RUN) pytest tests/test_docs -q
	$(UV_RUN) python tools/verify_claims.py --run $(RUNS_ROOT)/$(RUN_ID)

mutation-check:
	$(UV_RUN) python tools/mutation_check.py

provider-smoke:
	$(UV_RUN) python -m bist_predict.ingest.provider_smoke --ticker THYAO --days 21
