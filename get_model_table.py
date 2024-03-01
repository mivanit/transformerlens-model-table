from pathlib import Path
import warnings
import multiprocessing
from multiprocessing import Pool
from functools import partial
from copy import deepcopy

import pandas as pd
import tqdm
import yaml

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
from muutils.json_serialize import json_serialize

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
    "llama",
    "Llama-2",
    "bert",
    "tiny-stories",
    "stablelm",
    "bloom",
    "qwen",
    "mistral",
    "CodeLlama",
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
        include_cfg: bool = True,
        include_tensor_dims: bool = True,
        tensor_dims_fmt: str = "yaml",
    ) -> dict:
    # assumes the input is a default alias
    if model_name not in transformer_lens.loading.DEFAULT_MODEL_ALIASES:
        raise ValueError(f"Model name {model_name} not found in default aliases")

    official_name: str = MODEL_ALIASES_MAP.get(model_name, None)
    model_info: dict = {
        "name.default_alias": model_name,
        "name.official": official_name,
        "name.aliases": list(transformer_lens.loading.MODEL_ALIASES.get(official_name, [])),
        "model_type": None,
    }

    # Split the model name into parts
    parts: list[str] = model_name.split("-")

    # identify model type by known types
    for known_type in KNOWN_MODEL_TYPES:
        if known_type in model_name:
            model_info["model_type"] = known_type
            break

    # Search for model size
    param_count_from_name: str|None = None
    for part in parts:
        if (
            part[-1].lower() in ["m", "b", "k"]
            and part[:-1].replace(".", "", 1).isdigit()
        ):
            param_count_from_name = part
            break

    # update model info from config
    model_cfg: HookedTransformerConfig = get_pretrained_model_config(model_name)
    model_info.update({
        "name.from_cfg": model_cfg.model_name,
        "n_params.as_str": shorten_numerical_to_str(model_cfg.n_params),
        "n_params.as_int": model_cfg.n_params,
        "n_params.from_name": param_count_from_name,
        **{
            f"config.{attr}": getattr(model_cfg, attr)
            for attr in CONFIG_ATTRS_COPY
        },
    })



    # get the whole config
    if include_cfg:
        model_info["cfg"] = yaml.dump(
            json_serialize(model_cfg.to_dict()),
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )

    # get the model as a meta tensor
    if include_tensor_dims:
        try:
            model_cfg_copy: HookedTransformerConfig = deepcopy(model_cfg)
            model_cfg_copy.device = DEVICE
            model_cfg_copy.tokenizer_name = None
            model: HookedTransformer = HookedTransformer(model_cfg_copy, move_to_device=True)
            model_info["tensor_shapes.state_dict"] = condense_tensor_dict(model.state_dict(), return_format=tensor_dims_fmt)
            model_info["tensor_shapes.state_dict.raw__"] = condense_tensor_dict(model.state_dict(), return_format="dict")
            input_shape: tuple[int, int, int] = (847, model_cfg.n_ctx - 7)
            _, cache = model.run_with_cache(torch.empty(input_shape, dtype=torch.long, device=DEVICE))
            model_info["tensor_shapes.activation_cache"] = condense_tensor_dict(
                cache, 
                return_format=tensor_dims_fmt,
                dims_names_map={input_shape[0]: "batch", input_shape[1]: "seq_len"},
            )
            model_info["tensor_shapes.activation_cache.raw__"] = condense_tensor_dict(
                cache, 
                return_format="dict",
                dims_names_map={input_shape[0]: "batch", input_shape[1]: "seq_len"},
            )

        except Exception as e:
            warnings.warn(f"Failed to get tensor shapes for model {model_name}: {e}")
            return model_name, model_info

    return model_name, model_info


def safe_try_get_model_info(model_name: str, kwargs: dict|None = None) -> dict|None:
    if kwargs is None:
        kwargs = {}
    try:
        return get_model_info(model_name, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to get model info for {model_name}: {e}")
        return model_name, None


def make_model_table(
        verbose: bool, 
        allow_except: bool = False,
        parallelize: bool|int = True,
        **kwargs,
    ) -> pd.DataFrame:
    """make table of all models. kwargs passed to `get_model_info()`"""
    model_names: list[str] = list(transformer_lens.loading.DEFAULT_MODEL_ALIASES)
    model_data: list[tuple[str, dict|None]] = list()

    if parallelize:
        # parallel
        n_processes: int = parallelize if isinstance(parallelize, int) else multiprocessing.cpu_count()
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            # Use imap for ordered results, wrapped with tqdm for progress bar
            imap_results: list[dict|None] = list(tqdm.tqdm(
                pool.imap(
                    partial(safe_try_get_model_info, **kwargs),
                    model_names,
                ),
                total=len(model_names),
                desc="Loading model info",
                disable=not verbose,
            ))
        
        model_data = imap_results
    
    else:

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
                    model_data.append(None)

    failed_models: list[str] = [model_name for model_name, result in model_data if result is None]

    msg: str = f"Failed to get model info for {len(failed_models)}/{len(model_names)} models: {failed_models}"
    if not allow_except:
        if failed_models:
            raise ValueError(msg)        
    else:
        warnings.warn(msg)
    
    model_data_filtered: list[dict] = [result for _, result in model_data if result is not None]

    return pd.DataFrame(model_data_filtered)

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
        parallelize: bool|int = True,
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
        model_table: pd.DataFrame = make_model_table(verbose=verbose, parallelize=parallelize, **kwargs)
        if do_write:
            write_model_table(model_table, _MODEL_TABLE_PATH)
    else:
        model_table: pd.DataFrame = pd.read_json(_MODEL_TABLE_PATH, orient="records", lines=True)

    return model_table

if __name__ == "__main__":
    get_model_table(verbose=True, force_reload=True)