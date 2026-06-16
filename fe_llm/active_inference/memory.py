"""Memory candidate management for self-growth without online weight updates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .policy import ActionType
from .trace import InferenceTrace


@dataclass
class MemoryCandidate:
    text: str
    session_id: str | None
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "reason": self.reason,
            "metadata": self.metadata,
        }


def _char_bigrams(text: str) -> set[str]:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


class MemoryManager:
    """Records and recalls memory candidates; v1 never updates model weights online.

    v2 新增读回闭环：记下来的偏好/事实可以在后续轮次被 recall，
    注入信念假设并影响行动实现，使"成长"从只写不读变成可观测行为。
    """

    def __init__(self, candidate_path: str | None = os.path.join("data", "active_inference", "memory_candidates.jsonl")):
        self.candidate_path = candidate_path
        self.candidates: list[MemoryCandidate] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """启动时读回历史记忆候选，形成跨进程的最小持久记忆。"""

        if not self.candidate_path or not os.path.exists(self.candidate_path):
            return
        try:
            with open(self.candidate_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.candidates.append(
                        MemoryCandidate(
                            text=str(data.get("text", "")),
                            session_id=data.get("session_id"),
                            reason=str(data.get("reason", "")),
                            metadata=dict(data.get("metadata", {})),
                        )
                    )
        except Exception:
            # 记忆文件损坏不应阻断主闭环；从空记忆开始。
            self.candidates = []

    def recall(
        self,
        text: str,
        session_id: str | None = None,
        limit: int = 3,
        min_overlap: float = 0.3,
    ) -> list[MemoryCandidate]:
        """召回与当前观测相关的记忆。

        - 同一会话内的记忆（偏好/身份事实）默认全部相关，按时间倒序取最新若干条。
        - 跨会话记忆按字符 bigram 重合度筛选，避免无关记忆污染当前对话。
        """

        if not self.candidates:
            return []
        same_session: list[MemoryCandidate] = []
        cross_session: list[tuple[float, MemoryCandidate]] = []
        query_grams = _char_bigrams(text)
        for candidate in self.candidates:
            if session_id is not None and candidate.session_id == session_id:
                same_session.append(candidate)
                continue
            grams = _char_bigrams(candidate.text)
            if not grams or not query_grams:
                continue
            overlap = len(grams & query_grams) / max(1, min(len(grams), len(query_grams)))
            if overlap >= min_overlap:
                cross_session.append((overlap, candidate))
        cross_session.sort(key=lambda item: item[0], reverse=True)
        recalled = list(reversed(same_session))[:limit]
        for _, candidate in cross_session:
            if len(recalled) >= limit:
                break
            if candidate not in recalled:
                recalled.append(candidate)
        return recalled

    def audit_summary(self, confirm_threshold: int = 2, full_confidence_count: int = 3) -> list[dict[str, Any]]:
        """只读的成长审计视图：按文本聚合记忆候选，给出重复次数/置信/晋升状态。

        对应道易草案"穷则变"：单次出现只是候选，重复且稳定出现才晋升 confirmed
        （可进入长期记忆/离线再训练）。本方法不改持久化，仅供审计与离线评估。
        """
        from collections import defaultdict

        groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "sessions": set(), "reasons": set()})
        for candidate in self.candidates:
            key = candidate.text.strip()
            if not key:
                continue
            group = groups[key]
            group["count"] += 1
            if candidate.session_id:
                group["sessions"].add(candidate.session_id)
            if candidate.reason:
                group["reasons"].add(candidate.reason)

        summary: list[dict[str, Any]] = []
        for text, group in groups.items():
            count = int(group["count"])
            summary.append({
                "text": text,
                "count": count,
                "distinct_sessions": len(group["sessions"]),
                "confidence": round(min(1.0, count / max(full_confidence_count, 1)), 4),
                "status": "confirmed" if count >= confirm_threshold else "candidate",
                "reasons": sorted(group["reasons"]),
            })
        summary.sort(key=lambda item: item["count"], reverse=True)
        return summary

    def update_if_needed(self, trace: InferenceTrace) -> MemoryCandidate | None:
        if trace.selected_action.action_type != ActionType.UPDATE_MEMORY:
            return None
        candidate = MemoryCandidate(
            text=trace.observation.text,
            session_id=trace.observation.session_id,
            reason="User expressed a stable preference or identity fact.",
            metadata={
                "surprise": trace.surprise.to_dict(),
                "selected_action": trace.selected_action.action_type.value,
            },
        )
        self.candidates.append(candidate)
        if self.candidate_path:
            os.makedirs(os.path.dirname(self.candidate_path), exist_ok=True)
            with open(self.candidate_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
        return candidate

