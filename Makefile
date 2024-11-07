.DEFAULT_GOAL := all

.PHONY: all
all: format lint clean run

.PHONY: format
format:
	@echo "Running code formatting..."
	uv run black .
	uv run isort .
	uv run ruff check --fix .
	@echo "Code formatting completed."

.PHONY: lint
lint:
	@echo "Running code linting..."
	uv run black --check .
	uv run isort -c .
	uv run ruff check .
	@echo "Code linting completed."

.PHONY: run
run:
	@echo "Running the application..."
	uv run python src/main.py
	@echo "Application completed."

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf data
	rm -rf models
	rm -rf results
	@echo "Cleanup completed."
