HF_TOKEN=$(shell cat .hf-token)

BUILD_KWARGS ?= --verbose

dep:
	uv sync

build:
	HF_TOKEN=$(HF_TOKEN) uv run python get_model_table.py $(BUILD_KWARGS)

format:
	uv run python -m ruff check get_model_table.py
	uv run python -m ruff format get_model_table.py
	uv run python -m mypy get_model_table.py