HF_TOKEN=$(shell cat .hf-token)

BUILD_KWARGS ?=

dep:
	uv sync

build:
	HF_TOKEN=$(HF_TOKEN) uv run python get_model_table.py --verbose $(BUILD_KWARGS)