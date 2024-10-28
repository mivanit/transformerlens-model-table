HF_TOKEN=$(shell cat .hf-token)

dep:
	uv sync

build:
	HF_TOKEN=$(HF_TOKEN) uv run python get_model_table.py --verbose