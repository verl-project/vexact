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

import ast
from pathlib import Path
from types import ModuleType

import transformers.modeling_flash_attention_utils as fa_utils


def _load_patch_lazy_imports_for_fa4():
    script_path = Path(__file__).parent / "scripts" / "verify_logits_vs_native_hf.py"
    module = ast.parse(script_path.read_text(encoding="utf-8"))
    function_def = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_patch_lazy_imports_for_fa4"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function_def], type_ignores=[]), str(script_path), "exec"), namespace)
    return namespace["_patch_lazy_imports_for_fa4"]


def test_fa4_lazy_imports_patch_forwards_transformers_kwargs(monkeypatch):
    calls = []

    def original_lazy_imports(implementation, attention_wrapper=None, allow_all_kernels=False):
        calls.append((implementation, attention_wrapper, allow_all_kernels))
        return "ok"

    monkeypatch.setattr(fa_utils, "_lazy_imports", original_lazy_imports)

    patch_lazy_imports_for_fa4 = _load_patch_lazy_imports_for_fa4()
    patch_lazy_imports_for_fa4()

    assert fa_utils._lazy_imports("flash_attention_3", allow_all_kernels=True) == "ok"
    assert calls == [("flash_attention_3", None, True)]


def test_fa4_lazy_imports_patch_keeps_native_support(monkeypatch):
    def original_lazy_imports(implementation, attention_wrapper=None, allow_all_kernels=False):
        if implementation == "flash_attention_4":
            return "native"
        return "other"

    monkeypatch.setattr(fa_utils, "_lazy_imports", original_lazy_imports)

    patch_lazy_imports_for_fa4 = _load_patch_lazy_imports_for_fa4()
    assert patch_lazy_imports_for_fa4() is True
    assert fa_utils._lazy_imports is original_lazy_imports


def test_fa4_lazy_imports_patch_forwards_fa4_kwargs(monkeypatch):
    calls = []

    def original_lazy_imports(implementation, attention_wrapper=None, allow_all_kernels=False):
        calls.append((implementation, attention_wrapper, allow_all_kernels))
        return "ok"

    fake_flash_attn_cute = ModuleType("flash_attn.cute")
    fake_flash_attn_cute.flash_attn_func = object()
    fake_flash_attn_cute.flash_attn_varlen_func = object()

    import sys

    monkeypatch.setattr(fa_utils, "_lazy_imports", original_lazy_imports)
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake_flash_attn_cute)

    patch_lazy_imports_for_fa4 = _load_patch_lazy_imports_for_fa4()
    patch_lazy_imports_for_fa4()

    assert fa_utils._lazy_imports("flash_attention_4", allow_all_kernels=True) == "ok"
    implementation, attention_wrapper, allow_all_kernels = calls[0]
    assert implementation.flash_attn_func is fake_flash_attn_cute.flash_attn_func
    assert implementation.flash_attn_varlen_func is fake_flash_attn_cute.flash_attn_varlen_func
    assert attention_wrapper is None
    assert allow_all_kernels is True
