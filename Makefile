.PHONY: install train eval sanity test test-cov lint fmt clean

install:
	uv sync
	uv sync --all-extras
	uv sync --extra tracking

train:
	uv run python src/main.py $(ARGS)

sanity:
	uv run python scripts/run_sanity.py +experiment=sanity_cpu

eval:
	uv run python src/main.py run.mode=eval checkpoint.resume=$(CHECKPOINT)

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q

test-cov:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest --cov=src --cov-report=term-missing

lint:
	uv run ruff check src tests scripts/run_sanity.py

fmt:
	uv run ruff format src tests scripts/run_sanity.py

clean:
	rm -rf outputs .pytest_cache .ruff_cache .mypy_cache data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
