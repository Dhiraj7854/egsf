.PHONY: check test lint smoke e1
check: lint test
lint: ; ruff check .
test: ; pytest -q
smoke: ; python -m experiments.e1_ladder --config configs/bf.yaml --smoke
e1: ; python -m experiments.e1_ladder --config configs/d1_budget.yaml --seeds 0 1 2 3 4
