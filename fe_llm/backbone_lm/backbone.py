"""Delayed-import wrapper for pretrained causal language backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    """Hidden states and logits exposed to the FE-LLM mechanism layer."""

    hidden_states: torch.Tensor
    logits: torch.Tensor | None = None


class PretrainedBackbone(nn.Module):
    """Thin wrapper around a pretrained causal LM.

    这个类只定义 FE-LLM 需要的最小契约：底座输出 hidden states 和 logits。
    `transformers` 在 from_pretrained 内延迟导入，基础测试无需安装或下载模型。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        hidden_layer: int = -1,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_layer = hidden_layer
        if freeze:
            self.freeze()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        hidden_layer: int = -1,
        freeze: bool = True,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> "PretrainedBackbone":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "PretrainedBackbone.from_pretrained requires transformers. "
                "Install it only when running N1 backbone experiments."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **(tokenizer_kwargs or {}))
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **(model_kwargs or {}))
        return cls(model=model, tokenizer=tokenizer, hidden_layer=hidden_layer, freeze=freeze)

    def freeze(self) -> None:
        """Freeze backbone parameters; P1 only trains adapter/head."""

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode_texts(
        self,
        texts: list[str],
        *,
        max_length: int | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("encode_texts requires a tokenizer")
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BackboneOutput:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = output.hidden_states[self.hidden_layer]
        logits = getattr(output, "logits", None)
        return BackboneOutput(hidden_states=hidden_states, logits=logits)
