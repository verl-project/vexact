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
