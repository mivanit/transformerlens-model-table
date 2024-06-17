import os
import multiprocessing
import warnings
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import yaml
import tqdm
import torch

# muutils
from muutils.misc import shorten_numerical_to_str
from muutils.dictmagic import condense_tensor_dict
from muutils.json_serialize import json_serialize

# transformerlens
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

# forces everything to meta tensors
DEVICE: torch.device = torch.device("meta")
torch.set_default_device(DEVICE)

_MODEL_TABLE_PATH: Path = Path("docs/model_table.jsonl")

try:
    _hf_token = os.environ.get("HF_TOKEN", None)
    if not _hf_token.startswith("hf_"):
        raise ValueError("Invalid Hugging Face token")
except Exception as e:
    warnings.warn(f"Failed to get Hugging Face token -- mixtral models won't work\n{e}")

# manually defined known model types
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

# these will be copied as table columns
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

# modify certain values when printing config as yaml
CONFIG_VALUES_PROCESS: dict[str, Callable] = {
    "initializer_range": float,
}


def get_model_info(
    model_name: str,
    include_cfg: bool = True,
    include_tensor_dims: bool = True,
    tensor_dims_fmt: str = "yaml",
) -> dict:
    """get information about the model from the default alias model name

    # Parameters:
     - `model_name : str`
        the default alias model name
     - `include_cfg : bool`
        whether to include the model config as a yaml string
       (defaults to `True`)
     - `include_tensor_dims : bool`
        whether to include the model tensor shapes
       (defaults to `True`)
     - `tensor_dims_fmt : str`
        the format of the tensor shapes. one of "yaml", "json", "dict"
       (defaults to `"yaml"`)
    """
    # assumes the input is a default alias
    if model_name not in transformer_lens.loading.DEFAULT_MODEL_ALIASES:
        raise ValueError(f"Model name {model_name} not found in default aliases")

    # get the names and model types
    official_name: str = MODEL_ALIASES_MAP.get(model_name, None)
    model_info: dict = {
        "name.default_alias": model_name,
        "name.official": official_name,
        "name.aliases": list(
            transformer_lens.loading.MODEL_ALIASES.get(official_name, [])
        ),
        "model_type": None,
    }

    # Split the model name into parts
    parts: list[str] = model_name.split("-")

    # identify model type by known types
    for known_type in KNOWN_MODEL_TYPES:
        if known_type in model_name:
            model_info["model_type"] = known_type
            break

    # search for model size in name
    param_count_from_name: str | None = None
    for part in parts:
        if (
            part[-1].lower() in ["m", "b", "k"]
            and part[:-1].replace(".", "", 1).isdigit()
        ):
            param_count_from_name = part
            break

    # update model info from config
    model_cfg: HookedTransformerConfig = get_pretrained_model_config(model_name)
    model_info.update(
        {
            "name.from_cfg": model_cfg.model_name,
            "n_params.as_str": shorten_numerical_to_str(model_cfg.n_params),
            "n_params.as_int": model_cfg.n_params,
            "n_params.from_name": param_count_from_name,
            **{
                f"config.{attr}": getattr(model_cfg, attr) for attr in CONFIG_ATTRS_COPY
            },
        }
    )

    # put the whole config as yaml (for readability)
    if include_cfg:
        model_cfg_dict: dict = model_cfg.to_dict()
        # modify certain values to make them pretty-printable
        for key, func_process in CONFIG_VALUES_PROCESS.items():
            if key in model_cfg_dict:
                model_cfg_dict[key] = func_process(model_cfg_dict[key])
        # dump to yaml
        model_cfg_dict = json_serialize(model_cfg_dict)
        model_info["cfg.raw__"] = model_cfg_dict
        model_info["cfg"] = yaml.dump(
            model_cfg_dict,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )

    # get tensor shapes
    if include_tensor_dims:
        try:
            # copy the config, so we can modify it
            model_cfg_copy: HookedTransformerConfig = deepcopy(model_cfg)
            # set device to "meta" -- don't actually initialize the model with real tensors
            model_cfg_copy.device = DEVICE
            # don't need to download the tokenizer
            model_cfg_copy.tokenizer_name = None
            # init the fake model
            model: HookedTransformer = HookedTransformer(
                model_cfg_copy, move_to_device=True
            )
            # state dict
            model_info["tensor_shapes.state_dict"] = condense_tensor_dict(
                model.state_dict(), return_format=tensor_dims_fmt
            )
            model_info["tensor_shapes.state_dict.raw__"] = condense_tensor_dict(
                model.state_dict(), return_format="dict"
            )
            # input shape for activations -- "847"~="bat", subtract 7 for the context window to make it unique
            input_shape: tuple[int, int, int] = (847, model_cfg.n_ctx - 7)
            # why? to replace the batch and seq_len dims with "batch" and "seq_len" in the yaml
            dims_names_map: dict[int, str] = {
                input_shape[0]: "batch",
                input_shape[1]: "seq_len",
            }
            # run with cache to activation cache
            _, cache = model.run_with_cache(
                torch.empty(input_shape, dtype=torch.long, device=DEVICE)
            )
            # condense using muutils and store
            model_info["tensor_shapes.activation_cache"] = condense_tensor_dict(
                cache,
                return_format=tensor_dims_fmt,
                dims_names_map=dims_names_map,
            )
            model_info["tensor_shapes.activation_cache.raw__"] = condense_tensor_dict(
                cache,
                return_format="dict",
                dims_names_map=dims_names_map,
            )

        except Exception as e:
            warnings.warn(f"Failed to get tensor shapes for model {model_name}: {e}")
            return model_name, model_info

    return model_name, model_info


def safe_try_get_model_info(
    model_name: str, kwargs: dict | None = None
) -> dict | Exception:
    """for parallel processing, to catch exceptions and return the exception instead of raising them"""
    if kwargs is None:
        kwargs = {}
    try:
        return get_model_info(model_name, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to get model info for {model_name}: {e}")
        return model_name, e


def make_model_table(
    verbose: bool,
    allow_except: bool = False,
    parallelize: bool | int = True,
    **kwargs,
) -> pd.DataFrame:
    """make table of all models. kwargs passed to `get_model_info()`"""
    model_names: list[str] = list(transformer_lens.loading.DEFAULT_MODEL_ALIASES)
    model_data: list[tuple[str, dict | Exception]] = list()

    if parallelize:
        # parallel
        n_processes: int = (
            parallelize if isinstance(parallelize, int) else multiprocessing.cpu_count()
        )
        with Pool(processes=n_processes) as pool:
            # Use imap for ordered results, wrapped with tqdm for progress bar
            imap_results: list[dict | Exception] = list(
                tqdm.tqdm(
                    pool.imap(
                        partial(safe_try_get_model_info, **kwargs),
                        model_names,
                    ),
                    total=len(model_names),
                    desc="Loading model info",
                    disable=not verbose,
                )
            )

        model_data = imap_results

    else:
        # serial
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
                    if allow_except:
                        # warn and continue if we allow exceptions
                        warnings.warn(f"Failed to get model info for {model_name}: {e}")
                        model_data.append(e)
                    else:
                        # raise exception right away if we don't allow exceptions
                        # note that this differs from the parallel version, which will only except at the end
                        raise ValueError(
                            f"Failed to get model info for {model_name}"
                        ) from e

    # figure out what to do with failed models
    failed_models: dict[str, Exception] = {
        model_name: result
        for model_name, result in model_data
        if isinstance(result, Exception)
    }
    msg: str = (
        f"Failed to get model info for {len(failed_models)}/{len(model_names)} models: {failed_models}\n"
        + "\n".join(
            f"\t{model_name}: {expt}" for model_name, expt in failed_models.items()
        )
    )
    if not allow_except:
        if failed_models:
            # raise exception if we don't allow exceptions
            raise ValueError(msg + "\n\n" + "=" * 80 + "\n\n" + "NO DATA WRITTEN")
    else:
        warnings.warn(msg + "\n\n" + "-" * 80 + "\n\n" + "WRITING PARTIAL DATA")

    # filter out failed models if we allow exceptions
    model_data_filtered: list[dict] = [
        result for _, result in model_data if not isinstance(result, Exception)
    ]
    return pd.DataFrame(model_data_filtered)


OutputFormat = Literal["jsonl", "csv", "md"]


def write_model_table(
    model_table: pd.DataFrame,
    path: Path = _MODEL_TABLE_PATH,
    format: OutputFormat = "jsonl",
    include_TL_version: bool = True,
) -> None:
    """write the model table to disk in the specified format"""
    if include_TL_version:
        # get `transformer_lens` version
        tl_version: str = "unknown"
        try:
            from importlib.metadata import version, PackageNotFoundError

            tl_version = version("transformer_lens")
        except PackageNotFoundError as e:
            warnings.warn(
                f"Failed to get transformer_lens version: package not found\n{e}"
            )
        except Exception as e:
            warnings.warn(f"Failed to get transformer_lens version: {e}")

        with open(path.with_suffix(".version"), "w") as f:
            f.write(tl_version)

    match format:
        case "jsonl":
            model_table.to_json(path, orient="records", lines=True)
        case "csv":
            model_table.to_csv(path.with_suffix(".csv"), index=False)
        case "md":
            model_table.to_markdown(path.with_suffix(".md"), index=False)
        case _:
            raise KeyError(f"Invalid format: {format}")


def abridge_model_table(
    model_table: pd.DataFrame,
    max_mean_col_len: int = 100,
) -> pd.DataFrame:
    """remove columns which are too long from the model table, returning a new table

    primarily used to make the csv and md versions of the table readable
    """
    column_lengths: pd.Series = model_table.map(str).map(len).mean()
    columns_to_drop: list[str] = column_lengths[
        column_lengths > max_mean_col_len
    ].index.tolist()
    return model_table.drop(columns=columns_to_drop)


def get_model_table(
    verbose: bool = True,
    force_reload: bool = True,
    do_write: bool = True,
    parallelize: bool | int = True,
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
        # generate it from scratch
        model_table: pd.DataFrame = make_model_table(
            verbose=verbose, parallelize=parallelize, **kwargs
        )
        if do_write:
            # full data as jsonl
            write_model_table(model_table, _MODEL_TABLE_PATH, format="jsonl")
            # abridged data as csv, md
            abridged_table: pd.DataFrame = abridge_model_table(model_table)
            write_model_table(abridged_table, _MODEL_TABLE_PATH, format="csv")
            write_model_table(abridged_table, _MODEL_TABLE_PATH, format="md")
    else:
        # read the table from jsonl
        model_table: pd.DataFrame = pd.read_json(
            _MODEL_TABLE_PATH, orient="records", lines=True
        )

    return model_table


def main(**kwargs):
    """CLI for getting the model table and writing it to disk

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
    - `parallelize : bool | int`
        whether to parallelize the model info loading. if an int, specifies the number of processes
        (defaults to `True`)
    - `allow_except : bool`
        whether to allow exceptions when loading model info. If true, returns a table without rows for failed models
        (defaults to `False`)
    - `include_cfg : bool`
        whether to include the model config as a yaml string in the table (not included in csv or md, only jsonl)
        (defaults to `True`)
    - `include_tensor_dims : bool`
        whether to include the model tensor shapes (state dict, activation cache) in the table (not included in csv or md, only jsonl)
        (defaults to `True`)
    - `tensor_dims_fmt : Literal["yaml", "json", "dict"]`
        the format of the tensor shapes, passed to muutils.dictmagic.condense_tensor_dict
        (defaults to `"yaml"`)
    """
    get_model_table(**kwargs)


if __name__ == "__main__":
    import sys

    if "-h" in sys.argv or "--help" in sys.argv:
        print(main.__doc__)
        sys.exit(0)

    import fire

    fire.Fire(main)
