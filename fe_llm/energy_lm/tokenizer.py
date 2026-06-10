# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/tokenizer.py —— 字级分词器（含 [MASK]）
========================================================
最小能量对话模型用**字级**分词（中文一字一 token），足够小、足够透明。
特殊 token：
    [PAD] 填充   [MASK] 掩码（待坍缩去掩码的空位）
    [BOS] 回应起始   [EOS] 回应结束   [SEP] 分隔上文与回应

去掩码生成的本质：回应位置初始全是 [MASK]（最高能），逐步填成真字（能量下降）。
"""

from __future__ import annotations

PAD, MASK, BOS, EOS, SEP, UNK = "[PAD]", "[MASK]", "[BOS]", "[EOS]", "[SEP]", "[UNK]"
SPECIAL = [PAD, MASK, BOS, EOS, SEP, UNK]


class CharTokenizer:
    """字级分词器。"""

    def __init__(self, chars: list[str]):
        # 特殊 token 占据最前的 id
        self.id_to_tok = list(SPECIAL) + [c for c in chars if c not in SPECIAL]
        self.tok_to_id = {t: i for i, t in enumerate(self.id_to_tok)}
        self.pad_id = self.tok_to_id[PAD]
        self.mask_id = self.tok_to_id[MASK]
        self.bos_id = self.tok_to_id[BOS]
        self.eos_id = self.tok_to_id[EOS]
        self.sep_id = self.tok_to_id[SEP]
        self.unk_id = self.tok_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_tok)

    def encode(self, text: str) -> list[int]:
        return [self.tok_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            t = self.id_to_tok[i]
            if t in (PAD, BOS, EOS, SEP, MASK):
                continue
            out.append(t)
        return "".join(out)

    def is_special(self, i: int) -> bool:
        return i < len(SPECIAL)

    # ---- 存取 ----
    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_tok, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        import json
        with open(path, "r", encoding="utf-8") as f:
            id_to_tok = json.load(f)
        # 去掉特殊符再交给构造器（构造器会重新加）
        chars = [t for t in id_to_tok if t not in SPECIAL]
        return cls(chars)


def build_tokenizer() -> CharTokenizer:
    """从对话语料构建字表。"""
    from .corpus import all_chars
    return CharTokenizer(all_chars())


if __name__ == "__main__":
    tok = build_tokenizer()
    print(f"字表大小（含特殊符）：{tok.vocab_size}")
    print("编码 '你好'：", tok.encode("你好"))
    print("解码回去：", tok.decode(tok.encode("你好")))
