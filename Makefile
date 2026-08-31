.PHONY: check test lint check-imports install clean

# Run all checks: import isolation, then tests
check: check-imports test

install:
	uv sync --all-extras

# Ensure checker/ imports nothing from src/
check-imports:
	@echo "=== Checking checker/ import isolation ==="
	@python3 scripts/check_imports.py
	@echo "  checker import isolation: OK"

test:
	@echo "=== Running tests ==="
	uv run pytest tests/ -v --tb=short

lint:
	@echo "=== Lint (pyflakes) ==="
	uv run python -m pyflakes src/ checker/ campaigns/ tests/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/
