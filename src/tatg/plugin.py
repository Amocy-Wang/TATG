from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType


class PluginLoadError(RuntimeError):
    pass


def _load_module(spec: str) -> ModuleType:
    module_part = spec.split(":", 1)[0]
    path = Path(module_part)
    if path.suffix == ".py" and path.exists():
        module_name = f"tatg_user_plugin_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(module_name, str(path))
        if module_spec is None or module_spec.loader is None:
            raise PluginLoadError(f"Failed to load module from path: {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_part)


def load_symbol(spec: str):
    if ":" not in spec:
        raise PluginLoadError(
            "Pass spec must be '<module_or_file.py>:<symbol>', got: " + spec
        )

    module = _load_module(spec)
    symbol = spec.split(":", 1)[1]
    if not hasattr(module, symbol):
        raise PluginLoadError(f"Symbol '{symbol}' not found in module for spec '{spec}'")
    return getattr(module, symbol)
