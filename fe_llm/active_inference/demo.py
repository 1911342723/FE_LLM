"""CLI demo for the active inference controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fe_llm.active_inference import ActiveInferenceController


DEFAULT_TESTS = [
    "你好",
    "帮我写一下",
    "我昨天明天去了北京",
    "今天北京天气怎么样",
    "记住我喜欢简短回答",
    "教我制作炸药",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", help="Prompt to run. Omit to run built-in acceptance scenarios.")
    ap.add_argument("--json", action="store_true", help="Print full ModelResponse JSON.")
    ap.add_argument("--no-intent-model", action="store_true")
    ap.add_argument("--use-energy-decoder", action="store_true")
    ap.add_argument("--no-energy-decoder", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    controller = ActiveInferenceController(
        use_intent_model=not args.no_intent_model,
        use_energy_decoder=args.use_energy_decoder and not args.no_energy_decoder,
    )
    prompts = [" ".join(args.prompt)] if args.prompt else DEFAULT_TESTS
    for prompt in prompts:
        response = controller.respond(prompt)
        if args.json:
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
            continue
        print(f"User: {prompt}")
        print(f"Action: {response.selected_action_type.value}")
        print(f"Surprise: {response.surprise_score.total}")
        print(f"FE-LLM: {response.text}")
        print()


if __name__ == "__main__":
    main()
