.PHONY: run

run:
	uv run app.py

console:
	uv run textual console

run_dev: 
	uv run textual run --dev app.py

