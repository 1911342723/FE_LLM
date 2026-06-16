"""Utilities for intent-conditioned layer hooks.

P2c 的目标是把 intent residual 注入 decoder 层内，而不是只改最后一步
hidden。这里先提供与具体 HF 模型解耦的 hook 工具，后续 train/predict 再绑定
到 Qwen 的具体 layer path。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .residual_adapter import IntentResidualAdapter
from .types import IntentState


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve dotted module path, supporting numeric ModuleList indices."""

    current: Any = root
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{path!r} does not resolve to an nn.Module")
    return current


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError("Layer hook output must be a Tensor or tuple with hidden states first")


@dataclass
class IntentLayerHook:
    """Context manager that injects intent residual into selected layer outputs."""

    model: nn.Module
    layer_paths: list[str]
    adapter: IntentResidualAdapter
    intent_state: IntentState
    gamma: float = 1.0
    handles: list[Any] = field(default_factory=list, init=False)

    def _hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        adapted, _ = self.adapter(hidden, self.intent_state, gamma=self.gamma)
        return _replace_hidden(output, adapted)

    def __enter__(self) -> "IntentLayerHook":
        self.intent_state.validate()
        for path in self.layer_paths:
            module = resolve_module(self.model, path)
            self.handles.append(module.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
