# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from vexact.config import PPInfo
from vexact.models.register import register_models as _register_models
from vexact.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


_register_models()

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class StateDictIterator:
    filepaths: list[str]

    def __iter__(self) -> Generator[tuple[str, "torch.Tensor"], None, None]:
        for filepath in self.filepaths:
            if filepath.endswith(".safetensors"):
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        yield key, f.get_tensor(key)

            else:
                state_dict = torch.load(filepath, map_location="cpu", weights_only=True, mmap=True)
                for key in state_dict.keys():
                    yield key, state_dict[key]


def _load_state_dict(weights_path: str, **kwargs) -> list["StateDictIterator"]:
    """
    Loads (sharded) state dict in transformers' format.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return StateDictIterator([resolved_weight_file])

    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return StateDictIterator(shard_files)

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return StateDictIterator([resolved_weight_file])

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return StateDictIterator(shard_files)

    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


def load_weights_from_weight_path(model: nn.Module, model_config: PretrainedConfig, weights_path: str):
    weight_iterator = _load_state_dict(weights_path)
    load_weights_from_weight_iterator(model, model_config, weight_iterator)


def load_weights_from_weight_iterator(
    model: nn.Module,
    model_config: PretrainedConfig,
    weight_iterator: Iterable[tuple[str, torch.Tensor]],
):
    """
    All the model loading should go through this API to properly select the model loading method
    and determine whether to do weight tying.
    """
    # ignore the _tied_weights_keys in model instance when tie_word_embeddings in config is False
    tied_weight_keys = []
    if getattr(model_config, "tie_word_embeddings", False):
        tied_weight_keys = getattr(model, "_tied_weights_keys", list())

    if hasattr(model, "load_weights"):
        logger.info(f"Found custom load_weights method in model instance, {tied_weight_keys=}")
        model.load_weights(weight_iterator, tied_weight_keys)
    else:
        logger.info("Using default weight loading method")
        parameters = dict(model.named_parameters())
        embed_tokens_weight = None
        for full_name, loaded_weight in weight_iterator:
            if full_name == "model.embed_tokens.weight":
                embed_tokens_weight = loaded_weight

            if full_name in parameters:
                parameters[full_name].data.copy_(loaded_weight)

        for param_name in tied_weight_keys:
            if param_name in parameters:
                if "model.embed_tokens.weight" in parameters:
                    parameters[param_name].data = parameters["model.embed_tokens.weight"].data
                elif embed_tokens_weight is not None:
                    parameters[param_name].data.copy_(embed_tokens_weight)


def init_parameters(module: nn.Module, dtype: torch.dtype, device: torch.device):
    """
    If a `parameter` is on the `meta` device, then its parent
    `module` is the original module created by:

    ```python
    with torch.device("meta"):
        self.model: PreTrainedModel = AutoModel.from_config(...)
    ```
    """
    for name, param in module.named_parameters(recurse=False):
        if param.device == torch.device("meta"):
            new_param = nn.Parameter(torch.empty_like(param.data, dtype=dtype, device=device))
            setattr(module, name, new_param)
    for child in module.children():
        init_parameters(child, dtype, device)


def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
    """Try to evenly distribute layers across partitions.

    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all but the last
    partition. The last partition is excluded because it often contains an
    additional norm layer and we are attempting to balance compute.

    If `pp_size > 2` and the number of remaining layers is
    `0 < x <= pp_size - 2` then the remaining layers are evenly distributed
    across the middle partitions. The first and last partitions are excluded
    because they contain the input and output embeddings respectively and we
    are attempting to reduce maximum memory consumption across partitions.
    """
    layers_per_partition = num_hidden_layers // pp_size
    partitions = [layers_per_partition for _ in range(pp_size)]

    if remaining_layers := num_hidden_layers % pp_size:
        for i in range(2, remaining_layers + 2):
            partitions[-i] += 1
        logger.info(
            "Hidden layers were unevenly partitioned: [%s]. "
            # "This can be manually overridden using the "
            # "environment variable",  # TODO
            ",".join(str(p) for p in partitions),
        )

    start_layer = sum(partitions[:pp_rank])
    end_layer = start_layer + partitions[pp_rank]

    return (start_layer, end_layer)


@contextmanager
def init_on_device_without_buffers(device: torch.device):
    """
    A context manager under which models are initialized with all
    parameters on the specified device. However buffers are not
    initialized on specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
    """

    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = (
                param if param.device == device else param_cls(module._parameters[name].to(device), **kwargs)
            )

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


class TransformersForCausalLM(nn.Module):
    def __init__(self, model: nn.Module, lm_head: nn.Module, pp_info: PPInfo):
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self._pp_info = pp_info

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        intermediate_tensors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self._pp_info.pp_rank > 0:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors
        else:
            # First rank, we do not use intermediate_tensors
            assert input_ids is not None
            inputs_embeds = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        if self._pp_info.pp_rank != self._pp_info.pp_size - 1:
            return CausalLMOutputWithPast(
                loss=None,
                logits=None,
                past_key_values=None,
                hidden_states=outputs.last_hidden_state,
                attentions=None,
            )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )


class ModelCreator:
    def __init__(self, config: PretrainedConfig, model_path: str, device: torch.device, pp_info: PPInfo):
        self._pp_info = pp_info
        self._model_path = model_path
        self._config = config
        self._device = device
        with init_on_device_without_buffers("meta"):
            self._causal_model: PreTrainedModel = AutoModelForCausalLM.from_config(
                self._config,
            )
        self.model = self._causal_model.model

    def create_model(self):
        if torch.cuda.is_available():
            logger.info(
                f"Model load mem (before): "
                f"allocated={torch.cuda.memory_allocated() // (1024 * 1024)}MB "
                f"reserved={torch.cuda.memory_reserved() // (1024 * 1024)}MB"
            )
        if self._pp_info.pp_size > 1:
            self._apply_pp()

        with TorchMemorySaverAdapter.get_instance().region("weights", enable_cpu_backup=False):
            init_parameters(self._causal_model, self._config.dtype, self._device)

        load_weights_from_weight_path(self._causal_model, self._config, self._model_path)

        torch.cuda.empty_cache()
        self._causal_model.eval()
        if torch.cuda.is_available():
            logger.info(
                f"Model load mem (after): "
                f"allocated={torch.cuda.memory_allocated() // (1024 * 1024)}MB "
                f"reserved={torch.cuda.memory_reserved() // (1024 * 1024)}MB"
            )
        self._ensure_rotary_embeddings_on_device()
        for k, v in self._causal_model.named_buffers():
            logger.info(f"named_buffer [{k=}]: {v.dtype=}, {v.shape=}")
        # Note: Any patching to Causal LM instance should be reapplied here
        # so that PP Causal LM instance can get patched as well
        if self._pp_info.pp_size > 1:
            pp_model = TransformersForCausalLM(self.model, self._causal_model.lm_head, self._pp_info)
            if hasattr(self._causal_model, "_tied_weights_keys"):
                pp_model._tied_weights_keys = list(self._causal_model._tied_weights_keys)
            if hasattr(self._causal_model, "load_weights"):
                logger.info("[VEXACT] Register custom load_weights for PP model.")
                pp_model.load_weights = self._causal_model.load_weights
            return pp_model
        else:
            return self._causal_model

    def _ensure_rotary_embeddings_on_device(self):
        """Move RoPE buffers to target device to avoid host->device copies during CUDA graph capture."""
        rotary_emb = getattr(getattr(self._causal_model, "model", None), "rotary_emb", None)
        if rotary_emb is None:
            return
        if hasattr(rotary_emb, "inv_freq"):
            rotary_emb.inv_freq = rotary_emb.inv_freq.to(self._device)
        if hasattr(rotary_emb, "original_inv_freq"):
            rotary_emb.original_inv_freq = rotary_emb.original_inv_freq.to(self._device)

    def _apply_pp(self):
        """
        Apply the model's pipeline parallelization plan.
        """
        is_first_rank = self._pp_info.pp_rank == 0
        is_last_rank = self._pp_info.pp_rank == self._pp_info.pp_size - 1
        if not self.model.supports_pp_plan:
            raise ValueError(f"{type(self.model)} does not support pipeline parallel.")

        module_lists = []
        module_list_idx = None
        pp_plan = list(self.model._pp_plan.keys())
        for i, name in enumerate(pp_plan):
            if isinstance(getattr(self.model, name), nn.ModuleList):
                module_lists.append(name)
                module_list_idx = i

        if len(module_lists) > 1:
            raise ValueError(
                "Pipeline parallel of models with multiple `ModuleList`s in the base model are not supported yet!"
            )
        if module_list_idx is None:
            raise ValueError(f"Could not find `ModuleList` in {type(self.model)}")

        # Layers before module list
        for name in pp_plan[:module_list_idx]:
            if is_first_rank or (self._config.tie_word_embeddings and is_last_rank):
                continue
            setattr(self.model, name, PPMissingLayer())

        # Module list
        start_layer, end_layer = get_pp_indices(
            self._config.num_hidden_layers, self._pp_info.pp_rank, self._pp_info.pp_size
        )
        layers_name = pp_plan[module_list_idx]
        layers = getattr(self.model, layers_name)
        for i in range(len(layers)):
            if start_layer <= i and i < end_layer:
                continue
            layers[i] = PPMissingLayer()
            layers[i].attention_type = "full_attention"

        # Layers after module list
        for name in pp_plan[module_list_idx + 1 :]:
            # Modules that should be on last rank
            if self._pp_info.pp_rank != self._pp_info.pp_size - 1:
                setattr(self.model, name, PPMissingLayer())

        if self._pp_info.pp_rank != self._pp_info.pp_size - 1:
            self._causal_model.lm_head = PPMissingLayer()
