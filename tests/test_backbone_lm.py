from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from fe_llm.backbone_lm import (
    EnergyHead,
    IntentAdapter,
    IntentLayerHook,
    IntentLogitsAdapter,
    IntentResidualAdapter,
    PretrainedBackbone,
    resolve_module,
    select_hybrid_candidate,
)
from fe_llm.backbone_lm.slot_translation_p1_train import (
    build_arg_parser as build_p1_train_arg_parser,
    check_environment,
    load_pairs,
    main as p1_train_main,
    roll_intent_state,
    sequence_energy,
    translate_prompt as train_translate_prompt,
    unique_candidate_ids,
)
from fe_llm.backbone_lm.slot_translation_p15_train import (
    build_arg_parser as build_p15_train_arg_parser,
    main as p15_train_main,
)
from fe_llm.backbone_lm.slot_translation_p15_predict import (
    build_arg_parser as build_p15_predict_arg_parser,
    main as p15_predict_main,
)
from fe_llm.backbone_lm.slot_translation_p2_train import (
    build_arg_parser as build_p2_train_arg_parser,
    hard_negative_indices,
    main as p2_train_main,
)
from fe_llm.backbone_lm.slot_translation_p2_predict import (
    build_arg_parser as build_p2_predict_arg_parser,
    main as p2_predict_main,
)
from fe_llm.backbone_lm.slot_translation_p2c_predict import (
    build_arg_parser as build_p2c_predict_arg_parser,
    main as p2c_predict_main,
)
from fe_llm.backbone_lm.intent_adapter_diagnostic import (
    main as intent_diag_main,
    summarize_intents,
)
from fe_llm.backbone_lm.intent_adapter_train import (
    build_arg_parser as build_intent_train_arg_parser,
    contrastive_loss,
    global_spread_loss,
    main as intent_train_main,
    salience_entropy,
    slot_diversity_loss,
)
from fe_llm.backbone_lm.slot_translation_p1_eval import (
    P1Prediction,
    load_predictions,
    main as p1_eval_main,
    n1_verdict,
    score_prediction,
    summarize,
)
from fe_llm.backbone_lm.slot_translation_p1_predict import (
    build_translate_prompt,
    clean_generation,
    main as p1_predict_main,
    make_random_intent,
    prediction_record,
)
from fe_llm.backbone_lm.types import IntentState


class _FakeCausalOutput:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...], logits: torch.Tensor) -> None:
        self.hidden_states = hidden_states
        self.logits = logits


class _FakeCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 6)
        self.head = nn.Linear(6, 32)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embed(input_ids)
        return _FakeCausalOutput(hidden_states=(hidden * 0.5, hidden), logits=self.head(hidden))


class _TupleBlock(nn.Module):
    def forward(self, x):
        return (x + 1.0, {"meta": True})


class _HookedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False), _TupleBlock()])

    def forward(self, x):
        x = self.layers[0](x)
        x, meta = self.layers[1](x)
        return x, meta


class BackboneLmP1Tests(unittest.TestCase):
    def test_intent_adapter_outputs_structured_intent(self) -> None:
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, 16)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        adapter = IntentAdapter(hidden_size=16, intent_dim=8, n_slots=3, n_heads=4)

        state = adapter(hidden, attention_mask=attention_mask)

        self.assertEqual(state.global_intent.shape, (2, 8))
        self.assertEqual(state.intent_slots.shape, (2, 3, 8))
        self.assertEqual(state.slot_salience.shape, (2, 3))
        self.assertTrue(torch.allclose(state.slot_salience.sum(dim=-1), torch.ones(2)))

    def test_energy_head_reports_prefix_coverage_energy(self) -> None:
        torch.manual_seed(1)
        hidden = torch.randn(2, 4, 16)
        adapter = IntentAdapter(hidden_size=16, intent_dim=8, n_slots=3, n_heads=4)
        energy_head = EnergyHead(hidden_size=16, intent_dim=8)
        state = adapter(hidden)

        energies = energy_head(hidden, state)
        coverage = energies["coverage_energy"]

        self.assertEqual(energies["residual_energy"].shape, (2, 4))
        self.assertEqual(coverage.shape, (2, 4))
        # 覆盖能量使用前缀最小距离：生成越往后，已覆盖槽位的代价不应反弹。
        self.assertTrue(torch.all(coverage[:, 1:] <= coverage[:, :-1] + 1e-6))

    def test_hybrid_decode_can_choose_lower_energy_candidate(self) -> None:
        candidate_ids = torch.tensor([10, 20])
        log_probs = torch.tensor([0.0, -0.1])
        residual_energy = torch.tensor([1.0, 0.0])

        step = select_hybrid_candidate(candidate_ids, log_probs, residual_energy, alpha=1.0)

        self.assertEqual(step.prob_token_id, 10)
        self.assertEqual(step.energy_token_id, 20)
        self.assertEqual(step.token_id, 20)

    def test_pretrained_backbone_freezes_and_exposes_outputs(self) -> None:
        model = _FakeCausalModel()
        wrapper = PretrainedBackbone(model=model, hidden_layer=-1, freeze=True)
        input_ids = torch.tensor([[1, 2, 3]])

        output = wrapper(input_ids)

        self.assertFalse(wrapper.model.training)
        self.assertTrue(all(not param.requires_grad for param in wrapper.model.parameters()))
        self.assertEqual(output.hidden_states.shape, (1, 3, 6))
        self.assertEqual(output.logits.shape, (1, 3, 32))

    def test_intent_logits_adapter_scores_candidates(self) -> None:
        torch.manual_seed(3)
        state = IntentState(
            global_intent=torch.randn(2, 8),
            intent_slots=torch.randn(2, 3, 8),
            slot_salience=torch.softmax(torch.randn(2, 3), dim=-1),
        )
        adapter = IntentLogitsAdapter(hidden_size=16, intent_dim=8, adapter_dim=12)
        candidate_hidden = torch.randn(2, 4, 16)

        bias = adapter(candidate_hidden, state)
        one_bias = adapter(candidate_hidden[:, 0], state)

        self.assertEqual(bias.shape, (2, 4))
        self.assertEqual(one_bias.shape, (2,))

    def test_intent_residual_adapter_returns_bounded_delta(self) -> None:
        torch.manual_seed(4)
        state = IntentState(
            global_intent=torch.randn(2, 8),
            intent_slots=torch.randn(2, 3, 8),
            slot_salience=torch.softmax(torch.randn(2, 3), dim=-1),
        )
        adapter = IntentResidualAdapter(hidden_size=16, intent_dim=8, adapter_dim=12, max_delta_norm=0.5)
        hidden = torch.randn(2, 4, 16)

        adapted, delta = adapter(hidden, state, gamma=0.5)
        one_adapted, one_delta = adapter(hidden[:, 0], state)

        self.assertEqual(adapted.shape, hidden.shape)
        self.assertEqual(delta.shape, hidden.shape)
        self.assertLessEqual(float(delta.norm(dim=-1).max().detach()), 0.5001)
        self.assertEqual(one_adapted.shape, (2, 16))
        self.assertEqual(one_delta.shape, (2, 16))

    def test_intent_layer_hook_resolves_and_restores(self) -> None:
        torch.manual_seed(5)
        model = _HookedModel()
        state = IntentState(
            global_intent=torch.randn(2, 3),
            intent_slots=torch.randn(2, 2, 3),
            slot_salience=torch.softmax(torch.randn(2, 2), dim=-1),
        )
        adapter = IntentResidualAdapter(hidden_size=4, intent_dim=3, adapter_dim=6, max_delta_norm=0.5)
        x = torch.randn(2, 1, 4)

        baseline, meta = model(x)
        self.assertIs(resolve_module(model, "layers.1"), model.layers[1])
        with IntentLayerHook(model, ["layers.1"], adapter, state, gamma=1.0):
            hooked, hooked_meta = model(x)
        restored, restored_meta = model(x)

        self.assertEqual(meta, hooked_meta)
        self.assertFalse(torch.allclose(baseline, hooked))
        self.assertTrue(torch.allclose(baseline, restored))
        self.assertEqual(restored_meta, hooked_meta)

    def test_p1_load_pairs_skips_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"zh": "你好", "en": "hello"}, ensure_ascii=False) + "\n")
                f.write("{bad json}\n")
                f.write(json.dumps({"zh": "谢谢"}, ensure_ascii=False) + "\n")

            pairs = load_pairs(path)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].zh, "你好")
        self.assertEqual(pairs[0].en, "hello")

    def test_p1_negative_pairing_rolls_intent_state(self) -> None:
        state = IntentState(
            global_intent=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            intent_slots=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            slot_salience=torch.tensor([[1.0], [1.0]]),
        )

        neg = roll_intent_state(state)

        self.assertTrue(torch.equal(neg.global_intent[0], state.global_intent[1]))
        self.assertTrue(torch.equal(neg.intent_slots[1], state.intent_slots[0]))

    def test_p1_sequence_energy_returns_per_sample_energy(self) -> None:
        torch.manual_seed(2)
        hidden = torch.randn(2, 3, 16)
        adapter = IntentAdapter(hidden_size=16, intent_dim=8, n_slots=2, n_heads=4)
        energy_head = EnergyHead(hidden_size=16, intent_dim=8)
        state = adapter(hidden)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        energy = sequence_energy(energy_head, hidden, state, mask)

        self.assertEqual(energy.shape, (2,))
        self.assertTrue(torch.all(energy >= 0))

    def test_p1_train_main_defaults_to_dry_run(self) -> None:
        exit_code = p1_train_main([])

        self.assertEqual(exit_code, 0)

    def test_p1_train_check_environment_reports_data_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"zh": "你好", "en": "hello"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"zh": "谢谢", "en": "thanks"}, ensure_ascii=False) + "\n")
            parser = build_p1_train_arg_parser()
            args = parser.parse_args(["--check-env", "--train-path", path, "--device", "cpu"])

            result = check_environment(args)

        self.assertTrue(result["train_path_exists"])
        self.assertEqual(result["sample_pairs"], 2)
        self.assertTrue(result["has_min_pairs"])
        self.assertEqual(result["device"], "cpu")

    def test_p1_train_unique_candidate_ids_keeps_gold_first(self) -> None:
        self.assertEqual(unique_candidate_ids(3, [1, 3, 2, 1]), [3, 1, 2])

    def test_p1_train_candidate_loss_args_parse(self) -> None:
        parser = build_p1_train_arg_parser()
        args = parser.parse_args([
            "--candidate-loss-weight",
            "0.5",
            "--candidate-steps",
            "2",
            "--intent-contrast-weight",
            "0.25",
            "--seed",
            "7",
        ])

        self.assertEqual(args.candidate_loss_weight, 0.5)
        self.assertEqual(args.candidate_steps, 2)
        self.assertEqual(args.intent_contrast_weight, 0.25)
        self.assertEqual(args.seed, 7)
        self.assertEqual(train_translate_prompt(" 你好 "), "Chinese: 你好\nEnglish:")

    def test_p1_eval_load_predictions_skips_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pred.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"group": "A", "zh": "你好", "ref": "hello", "pred": "hello"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write("{bad json}\n")
                f.write(json.dumps({"group": "B", "zh": "谢谢", "ref": "thanks"}, ensure_ascii=False) + "\n")

            rows = load_predictions(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].group, "A")

    def test_p1_eval_summary_and_pass_verdict(self) -> None:
        rows = [
            score_prediction(P1Prediction("A", "你好", "hello world", "hello")),
            score_prediction(P1Prediction("B", "你好", "hello world", "hello world")),
            score_prediction(P1Prediction("C", "你好", "hello world", "goodbye")),
        ]

        summary = summarize(rows)
        verdict = n1_verdict(summary)

        self.assertEqual(summary["B"]["mean_word_f1"], 1.0)
        self.assertTrue(verdict.startswith("PASS"))

    def test_p1_eval_requires_b_to_beat_controls(self) -> None:
        rows = [
            score_prediction(P1Prediction("A", "你好", "hello world", "hello world")),
            score_prediction(P1Prediction("B", "你好", "hello world", "hello world")),
            score_prediction(P1Prediction("C", "你好", "hello world", "bad output")),
        ]

        verdict = n1_verdict(summarize(rows))

        self.assertIn("does not beat A/C", verdict)

    def test_p1_eval_main_defaults_to_dry_run(self) -> None:
        exit_code = p1_eval_main([])

        self.assertEqual(exit_code, 0)

    def test_p1_predict_helpers_build_prompt_and_clean_generation(self) -> None:
        self.assertEqual(build_translate_prompt(" 你好 "), "Chinese: 你好\nEnglish:")
        self.assertEqual(clean_generation(" hello world\nChinese: next"), "hello world")

    def test_p1_predict_random_intent_keeps_shapes(self) -> None:
        state = IntentState(
            global_intent=torch.zeros(1, 4),
            intent_slots=torch.zeros(1, 2, 4),
            slot_salience=torch.tensor([[0.8, 0.2]]),
        )

        random_state = make_random_intent(state, seed=123)

        self.assertEqual(random_state.global_intent.shape, state.global_intent.shape)
        self.assertEqual(random_state.intent_slots.shape, state.intent_slots.shape)
        self.assertTrue(torch.allclose(random_state.slot_salience, torch.tensor([[0.5, 0.5]])))

    def test_p1_predict_record_keeps_optional_trace_fields(self) -> None:
        pair = type("Pair", (), {"zh": "你好", "en": "hello"})()
        record = prediction_record(
            "B",
            pair,
            "hello\nextra",
            {"residual_start": 2.0, "residual_end": 1.0, "disagreement_rate": 0.25},
        )

        self.assertEqual(record["group"], "B")
        self.assertEqual(record["pred"], "hello")
        self.assertEqual(record["residual_end"], 1.0)
        self.assertEqual(record["disagreement_rate"], 0.25)

    def test_p1_predict_main_defaults_to_dry_run(self) -> None:
        exit_code = p1_predict_main([])

        self.assertEqual(exit_code, 0)

    def test_p15_train_args_and_dry_run(self) -> None:
        parser = build_p15_train_arg_parser()
        args = parser.parse_args([
            "--candidate-steps",
            "2",
            "--intent-contrast-weight",
            "0.25",
            "--ckpt-name",
            "custom.pt",
        ])

        self.assertEqual(args.candidate_steps, 2)
        self.assertEqual(args.intent_contrast_weight, 0.25)
        self.assertEqual(args.ckpt_name, "custom.pt")
        self.assertEqual(p15_train_main([]), 0)

    def test_p15_predict_args_and_dry_run(self) -> None:
        parser = build_p15_predict_arg_parser()
        args = parser.parse_args(["--beta", "0.7", "--top-k", "4"])

        self.assertEqual(args.beta, 0.7)
        self.assertEqual(args.top_k, 4)
        self.assertEqual(p15_predict_main([]), 0)

    def test_p2_train_args_and_dry_run(self) -> None:
        parser = build_p2_train_arg_parser()
        args = parser.parse_args([
            "--gamma",
            "0.5",
            "--delta-norm-weight",
            "0.02",
            "--negative-mode",
            "hard",
            "--intent-ckpt-path",
            "intent.pt",
        ])

        self.assertEqual(args.gamma, 0.5)
        self.assertEqual(args.delta_norm_weight, 0.02)
        self.assertEqual(args.negative_mode, "hard")
        self.assertEqual(args.intent_ckpt_path, "intent.pt")
        self.assertEqual(p2_train_main([]), 0)

    def test_p2_hard_negative_indices_pick_nearest_non_self(self) -> None:
        state = IntentState(
            global_intent=torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]]),
            intent_slots=torch.zeros(3, 1, 2),
            slot_salience=torch.ones(3, 1),
        )

        self.assertEqual(hard_negative_indices(state), [1, 0, 1])

    def test_p2_predict_args_and_dry_run(self) -> None:
        parser = build_p2_predict_arg_parser()
        args = parser.parse_args([
            "--gamma",
            "0.75",
            "--top-k",
            "4",
            "--control-mode",
            "mismatch",
            "--intent-ckpt-path",
            "intent.pt",
        ])

        self.assertEqual(args.gamma, 0.75)
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.control_mode, "mismatch")
        self.assertEqual(args.intent_ckpt_path, "intent.pt")
        self.assertEqual(p2_predict_main([]), 0)

    def test_p2c_predict_args_and_dry_run(self) -> None:
        parser = build_p2c_predict_arg_parser()
        args = parser.parse_args(["--layer-path", "model.layers.1", "--gamma", "0.5"])

        self.assertEqual(args.layer_path, "model.layers.1")
        self.assertEqual(args.gamma, 0.5)
        self.assertEqual(p2c_predict_main([]), 0)

    def test_intent_diagnostic_summary_and_dry_run(self) -> None:
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        salience = torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.9, 0.1]])

        summary = summarize_intents(z, salience)

        self.assertEqual(summary["n"], 3)
        self.assertIn("mean_offdiag_cosine", summary)
        self.assertEqual(intent_diag_main([]), 0)

    def test_intent_adapter_train_helpers_and_dry_run(self) -> None:
        parser = build_intent_train_arg_parser()
        args = parser.parse_args(["--entropy-weight", "0.1", "--slot-div-weight", "0.2", "--source-spread-weight", "1.0"])
        z = torch.eye(3)
        slots = torch.randn(2, 3, 4)
        salience = torch.softmax(torch.randn(2, 3), dim=-1)

        self.assertEqual(args.entropy_weight, 0.1)
        self.assertEqual(args.source_spread_weight, 1.0)
        self.assertGreaterEqual(float(contrastive_loss(z, z, 0.07).detach()), 0.0)
        self.assertGreaterEqual(float(global_spread_loss(z).detach()), 0.0)
        self.assertGreaterEqual(float(slot_diversity_loss(slots).detach()), 0.0)
        self.assertGreaterEqual(float(salience_entropy(salience).detach()), 0.0)
        self.assertEqual(intent_train_main([]), 0)


if __name__ == "__main__":
    unittest.main()
