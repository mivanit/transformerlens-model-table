from pathlib import Path

import pandas as pd
import tqdm
import transformer_lens
from muutils.misc import shorten_numerical_to_str
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import (
    get_pretrained_model_config
)

_MODEL_TABLE_PATH: Path = Path("data/model_table.jsonl")

KNOWN_MODEL_TYPES: list[str] = [
    "gpt2",
    "distillgpt2",
    "opt",
    "gpt-neo",
    "gpt-j",
    "gpt-neox",
    "stanford-gpt2",
    "pythia",
    "solu",
    "gelu",
    "attn-only",
    "redwood_attn",
    "llama",
    "Llama-2",
    "othello-gpt",
    "bert",
    "tiny-stories",
    "stablelm",
    "bloom",
    "santacoder",
]

MODEL_ALIASES_MAP: dict[str, str] = transformer_lens.loading.make_model_alias_map()

CONFIG_ATTRS_COPY: list[str] = [
    "n_params",
    "n_layers",
    "n_heads",
    "d_model",
    "d_vocab",
    "act_fn",
    "positional_embedding_type",
    "parallel_attn_mlp",
    "original_architecture",
]

def get_model_info(model_name: str, include_cfg: bool = False) -> dict:
    # assumes the input is a default alias
    if model_name not in transformer_lens.loading.DEFAULT_MODEL_ALIASES:
        raise ValueError(f"Model name {model_name} not found in default aliases")

    # output
    model_info: dict = dict(
        default_alias=model_name,
        official_name=MODEL_ALIASES_MAP.get(model_name, None),
        model_size_info=None,
        model_type=None,
    )

    # Split the model name into parts
    parts: list[str] = model_name.split("-")

    # Search for model size
    for part in parts:
        if (
            part[-1].lower() in ["m", "b", "k"]
            and part[:-1].replace(".", "", 1).isdigit()
        ):
            model_info["model_size_info"] = part
            break

    # identify model type by known types
    for known_type in KNOWN_MODEL_TYPES:
        if known_type in model_name:
            model_info["model_type"] = known_type
            break

    # get the config
    model_cfg: HookedTransformerConfig = get_pretrained_model_config(model_name)
    if include_cfg:
        model_info["cfg"] = model_cfg

    # update model info from config
    model_info.update(dict(
        cfg_model_name=model_cfg.model_name,
        n_params_str=shorten_numerical_to_str(model_cfg.n_params),
        **{
            attr: getattr(model_cfg, attr)
            for attr in CONFIG_ATTRS_COPY
        },
    ))

    return model_info



def make_model_table(verbose: bool) -> pd.DataFrame:
    model_data: list[dict] = list()

    for model_name in tqdm.tqdm(
        transformer_lens.loading.DEFAULT_MODEL_ALIASES,
        desc="Loading model info",
        disable=not verbose,
    ):
        model_data.append(get_model_info(model_name))

    model_table: pd.DataFrame = pd.DataFrame(model_data)

    return model_table


def write_model_table(model_table: pd.DataFrame, path: Path = _MODEL_TABLE_PATH) -> None:
    # to jsonlines
    model_table.to_json(path, orient="records", lines=True)
    # to csv
    model_table.to_csv(path.with_suffix(".csv"), index=False)
    # to markdown table
    model_table.to_markdown(path.with_suffix(".md"), index=False)


def get_model_table(verbose: bool = True, force_reload: bool = True) -> pd.DataFrame:
    if not _MODEL_TABLE_PATH.exists() or force_reload:
        model_table: pd.DataFrame = make_model_table(verbose)
        write_model_table(model_table, _MODEL_TABLE_PATH)
    else:
        model_table: pd.DataFrame = pd.read_json(_MODEL_TABLE_PATH, orient="records", lines=True)

    return model_table

if __name__ == "__main__":
    get_model_table(verbose=True, force_reload=True)