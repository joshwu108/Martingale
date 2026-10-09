.PHONY: check test lint typecheck check-imports install clean release

# Run all checks: import isolation, then tests
check: check-imports lint typecheck test

install:
	uv sync --all-extras

# Build and validate release artifacts locally; publishing is a separate manual step.
release:
	uv build
	uv run --with twine twine check dist/*

# Ensure checker/ imports nothing from src/
check-imports:
	@echo "=== Checking checker/ import isolation ==="
	@python3 scripts/check_imports.py
	@echo "  checker import isolation: OK"

test:
	@echo "=== Running tests ==="
	uv run pytest tests/ -v --tb=short

lint:
	@echo "=== Lint (ruff) ==="
	uv run ruff check src/ checker/ campaigns/ tests/ benchmarks/

typecheck:
	@echo "=== Type check (mypy: record, diagnostics, exact, checker) ==="
	uv run mypy

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/
