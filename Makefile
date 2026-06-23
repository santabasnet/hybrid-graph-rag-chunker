.PHONY: start sync run format clean

# Install/sync dependencies into the uv-managed virtual environment.
sync:
	uv sync --dev

# Run the console app (auto-syncs deps first via `uv run`).
start: sync
	uv run python main.py

# Equivalent direct invocation, useful for debugging imports.
run:
	uv run python main.py

# Format all Python source files (80-char line cap — configured in pyproject.toml).
format:
	uv run ruff format src/ main.py tests/

clean:
	rm -rf .venv chunked_output.json
