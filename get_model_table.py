from pathlib import Path
import warnings

import pandas as pd
import tqdm

# pytorch
import torch

# transformerlens
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import (
    get_pretrained_model_config
)

# muutils
from muutils.misc import shorten_numerical_to_str
from muutils.dictmagic import condense_tensor_dict

# forces everything to meta tensors
DEVICE: torch.device = torch.device("meta")
torch.set_default_device(DEVICE)

_MODEL_TABLE_PATH: Path = Path("docs/model_table.jsonl")

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
    "normalization_type",
]

def get_model_info(
        model_name: str, 
        include_cfg: bool = False,
        include_tensor_dims: bool = True,
        tensor_dims_fmt: str = "yaml",
    ) -> dict:
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

    # get the model as a meta tensor
    if include_tensor_dims:
        try:
            model_cfg.device = DEVICE
            model: HookedTransformer = HookedTransformer(model_cfg, move_to_device=True)
            model_info["state_dict"] = condense_tensor_dict(model.state_dict(), return_format=tensor_dims_fmt)
            input_shape: tuple[int, int, int] = (847, model_cfg.n_ctx - 7)
            _, cache = model.run_with_cache(torch.empty(input_shape, dtype=torch.long, device=DEVICE))
            model_info["activation_cache"] = condense_tensor_dict(
                cache, 
                return_format=tensor_dims_fmt,
                dims_names_map={input_shape[0]: "batch", input_shape[1]: "seq_len"},
            )
        except Exception as e:
            warnings.warn(f"Failed to get tensor shapes for model {model_name}: {e}")
            for k in ["state_dict", "activation_cache"]:
                if k not in model_info:
                    model_info[k] = None

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



def make_model_table(verbose: bool, **kwargs) -> pd.DataFrame:
    """make table of all models. kwargs passed to `get_model_info()`"""
    model_data: list[dict] = list()

    with tqdm.tqdm(
        transformer_lens.loading.DEFAULT_MODEL_ALIASES,
        desc="Loading model info",
        disable=not verbose,
    ) as pbar:
        for model_name in pbar:
            pbar.set_postfix_str(f"model: {model_name}")
            try:
                model_data.append(get_model_info(model_name, **kwargs))
            except Exception as e:
                warnings.warn(f"Failed to get model info for {model_name}: {e}")

    model_table: pd.DataFrame = pd.DataFrame(model_data)

    return model_table


def write_model_table(model_table: pd.DataFrame, path: Path = _MODEL_TABLE_PATH) -> None:
    # to jsonlines
    model_table.to_json(path, orient="records", lines=True)
    # to csv
    model_table.to_csv(path.with_suffix(".csv"), index=False)
    # to markdown table
    model_table.to_markdown(path.with_suffix(".md"), index=False)


def get_model_table(
        verbose: bool = True, 
        force_reload: bool = True,
        do_write: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
    """get the model table either by generating or reading from jsonl file

    # Parameters:
     - `verbose : bool`   
        whether to show progress bar
       (defaults to `True`)
     - `force_reload : bool`   
        force creating the table from scratch, even if file exists
       (defaults to `True`)
     - `do_write : bool`   
        whether to write the table to disk, if generating
       (defaults to `True`)
     - `**kwargs`
        eventually passed to `get_model_info()`
    
    # Returns:
     - `pd.DataFrame` 
        the model table. rows are models, columns are model attributes
    """    
    
    if not _MODEL_TABLE_PATH.exists() or force_reload:
        model_table: pd.DataFrame = make_model_table(verbose)
        if do_write:
            write_model_table(model_table, _MODEL_TABLE_PATH)
    else:
        model_table: pd.DataFrame = pd.read_json(_MODEL_TABLE_PATH, orient="records", lines=True)

    return model_table

if __name__ == "__main__":
    get_model_table(verbose=True, force_reload=True)