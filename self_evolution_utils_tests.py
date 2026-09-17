from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent_runner
import agent_proposer
from run_self_evolution_train import _parse_args, _snapshot
from skill_case_store import append_skill_case, compact_compressed_case_bodies, iter_skill_cases, read_skill_case
from skill_compact import deduplicate_skill_file, promote_effective_cases
from self_evolution_utils import (
    build_manifest,
    checkpoint_names,
    collect_feature_result,
    copy_skills_checkpoint,
    feature_batches,
    hash_tree,
    make_main_experiment_cmd,
    write_json,
)


class SelfEvolutionUtilsTest(unittest.TestCase):
    def test_top_k_normalization_falls_back_for_invalid_llm_values(self) -> None:
        self.assertEqual(agent_runner._normalize_top_k(None), 30)
        self.assertEqual(agent_runner._normalize_top_k(""), 30)
        self.assertEqual(agent_runner._normalize_top_k("invalid"), 30)
        self.assertEqual(agent_runner._normalize_top_k(0), 30)
        self.assertEqual(agent_runner._normalize_top_k("50"), 50)

    def test_manifest_has_expected_counts_and_no_overlap(self) -> None:
        manifest = build_manifest(layers=[0, 6, 12, 18, 24], train_per_layer=20, val_per_layer=10)
        for layer in manifest["layers"]:
            train = manifest["train_features"][str(layer)]
            val = manifest["validation_features"][str(layer)]
            self.assertEqual(len(train), 20)
            self.assertEqual(len(val), 10)
            self.assertFalse(set(train) & set(val))

    def test_checkpoint_cadence(self) -> None:
        self.assertEqual(
            checkpoint_names(100, 10),
            [
                "ckpt_000",
                "ckpt_010",
                "ckpt_020",
                "ckpt_030",
                "ckpt_040",
                "ckpt_050",
                "ckpt_060",
                "ckpt_070",
                "ckpt_080",
                "ckpt_090",
                "ckpt_100",
            ],
        )

    def test_checkpoint_cadence_supports_continuation_offset(self) -> None:
        self.assertEqual(
            checkpoint_names(80, 20, offset=120),
            ["ckpt_120", "ckpt_140", "ckpt_160", "ckpt_180", "ckpt_200"],
        )

    def test_existing_offset_checkpoint_keeps_original_index_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index_path = root / "checkpoints" / "index.json"
            checkpoint_dir = index_path.parent / "ckpt_120"
            checkpoint_dir.mkdir(parents=True)
            prior = {
                "name": "ckpt_120",
                "processed_features": 120,
                "path": str(checkpoint_dir),
                "hash": "original-hash",
                "created_at": "original-time",
                "skill_maintenance": {"status": "original"},
            }
            index = {"checkpoints": [dict(prior)]}
            args = SimpleNamespace(force=False, skip_skill_compact=False)

            with patch("run_self_evolution_train._run_skill_maintenance") as maintenance:
                _snapshot(
                    args=args,
                    index=index,
                    index_path=index_path,
                    name="ckpt_120",
                    processed_features=120,
                )

            maintenance.assert_not_called()
            self.assertEqual(index["checkpoints"], [prior])
            self.assertEqual(json.loads(index_path.read_text(encoding="utf-8"))["checkpoints"], [prior])

    def test_command_has_one_feature_per_layer(self) -> None:
        batch = {0: 10, 6: 20, 12: 30, 18: 40, 24: 50}
        cmd = make_main_experiment_cmd(
            timestamp="ts",
            batch=batch,
            feature_workers=5,
            skills_dir=Path("skills"),
            disable_skill_learning=True,
        )
        lf_idx = cmd.index("--layer-feature-ids")
        workers_idx = cmd.index("--feature-workers")
        self.assertEqual(cmd[lf_idx + 1 : workers_idx], ["0:10", "6:20", "12:30", "18:40", "24:50"])
        self.assertIn("--disable-skill-learning", cmd)
        self.assertEqual(cmd[cmd.index("--feature-workers") + 1], "5")

    def test_command_supports_multiple_features_per_layer(self) -> None:
        cmd = make_main_experiment_cmd(
            timestamp="ts",
            batch={0: [10, 11], 6: [20, 21]},
            feature_workers=10,
        )
        lf_idx = cmd.index("--layer-feature-ids")
        workers_idx = cmd.index("--feature-workers")
        self.assertEqual(cmd[lf_idx + 1 : workers_idx], ["0:10,11", "6:20,21"])
        self.assertEqual(cmd[cmd.index("--feature-workers") + 1], "10")

    def test_command_can_explicitly_disable_main_skill_maintenance(self) -> None:
        cmd = make_main_experiment_cmd(
            timestamp="ts",
            batch={0: [10]},
            skill_maintenance_mode="off",
        )
        mode_idx = cmd.index("--skill-maintenance")
        self.assertEqual(cmd[mode_idx + 1], "off")

    def test_self_evolution_rejects_main_auto_maintenance_passthrough(self) -> None:
        with patch(
            "sys.argv",
            ["run_self_evolution_train.py", "--skill-maintenance", "auto"],
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                _parse_args()
        self.assertEqual(raised.exception.code, 2)

    def test_checkpoint_hash_and_copy_ignore_runtime_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "skills"
            source.mkdir()
            (source / "skill_input.md").write_text("content\n", encoding="utf-8")
            before = hash_tree(source)
            (source / ".maintenance.lock").write_text("runtime\n", encoding="utf-8")
            self.assertEqual(hash_tree(source), before)

            destination = root / "checkpoint"
            copy_skills_checkpoint(source, destination)
            self.assertFalse((destination / ".maintenance.lock").exists())
            self.assertEqual(hash_tree(destination), before)

    def test_collect_fake_trace_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            logs_root = Path(td)
            ts_dir = logs_root / "layer-0" / "feature-30" / "eval_ts"
            trace = {
                "input_round": {
                    "eval": {
                        "per_hypothesis": [
                            {"score_non_zero_rate": 0.2, "score_boundary_non_activation_rate": 0.9},
                            {"score_non_zero_rate": 0.9, "score_boundary_non_activation_rate": 0.85},
                        ]
                    }
                },
                "chain": {
                    "best_pair_idx": 1,
                    "pairs": [{"idx": 1, "chain_judge_score": 4, "output_score": 0.7}],
                },
                "meta": {"token_cost": {"total": {"total_tokens": 123}}},
            }
            write_json(ts_dir / "trace.json", trace)
            write_json(ts_dir / "agent_loop_summary.json", {"status": "complete", "rounds_run": 2})
            row = collect_feature_result(
                layer=0,
                feature_id=30,
                timestamp="eval_ts",
                checkpoint="ckpt_010",
                logs_root=logs_root,
            )
            self.assertTrue(row["gate1_pass"])
            self.assertTrue(row["gate2_pass"])
            self.assertTrue(row["gate3_pass"])
            self.assertTrue(row["all_gates_pass"])
            self.assertEqual(row["rounds_run"], 2)
            self.assertEqual(row["total_tokens"], 123)


    def test_skill_case_store_reads_case_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td)
            case_id = append_skill_case(
                skills_dir,
                side="output",
                heading="## [Case][output][case: output-demo] Demo",
                body="**When:** demo\n\n**Fix:** demo",
                meta={
                    "case_id": "output-demo",
                    "side": "output",
                    "usage": {
                        "use_count": 0,
                        "effective_use_count": 0,
                        "compressed": False,
                    },
                },
            )
            self.assertEqual(case_id, "output-demo")
            content = read_skill_case(skills_dir, "output-demo")
            self.assertIsNotNone(content)
            cases = iter_skill_cases(skills_dir, side="output")
            self.assertEqual(len(cases), 1)
            self.assertNotIn("read_count", cases[0].meta["usage"])
            self.assertEqual(cases[0].meta["usage"]["use_count"], 0)
            self.assertFalse((skills_dir / "skill_usage.jsonl").exists())
            self.assertFalse((skills_dir / "skill_cases_index.json").exists())

    def test_compressed_case_archive_lives_outside_skills(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills_dir = root / "skills"
            archive_dir = root / "analysis_output" / "self_evolution" / "exp1" / "skill_case_archive"
            append_skill_case(
                skills_dir,
                side="output",
                heading="## [Case][output][case: output-archive-demo] Demo",
                body="**When:** original body\n\n**Fix:** original fix",
                meta={
                    "case_id": "output-archive-demo",
                    "side": "output",
                    "usage": {
                        "use_count": 0,
                        "effective_use_count": 2,
                        "compressed": True,
                        "compressed_target": "skill_output.md",
                    },
                },
            )
            changed = compact_compressed_case_bodies(skills_dir, archive_dir=archive_dir)
            self.assertEqual(changed, ["skill_output_cases.md"])
            self.assertTrue((archive_dir / "skill_output_cases_archive.md").exists())
            self.assertFalse((skills_dir / "skill_output_cases_archive.md").exists())
            active = (skills_dir / "skill_output_cases.md").read_text(encoding="utf-8")
            archived = (archive_dir / "skill_output_cases_archive.md").read_text(encoding="utf-8")
            self.assertNotIn("output-archive-demo", active)
            self.assertEqual(iter_skill_cases(skills_dir, side="output"), [])
            self.assertIn("original fix", archived)

    def test_promotion_compresses_supporting_cases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills_dir = root / "skills"
            archive_dir = root / "analysis_output" / "self_evolution" / "exp1" / "skill_case_archive"
            (skills_dir / "skill_output.md").parent.mkdir(parents=True, exist_ok=True)
            (skills_dir / "skill_output.md").write_text("", encoding="utf-8")
            append_skill_case(
                skills_dir,
                side="output",
                heading="## [Case][output][case: output-primary] Primary",
                body="primary body",
                meta={
                    "case_id": "output-primary",
                    "side": "output",
                    "usage": {"effective_use_count": 2, "compressed": False},
                },
            )
            append_skill_case(
                skills_dir,
                side="output",
                heading="## [Case][output][case: output-ref] Reference",
                body="reference body",
                meta={
                    "case_id": "output-ref",
                    "side": "output",
                    "usage": {"effective_use_count": 0, "compressed": False},
                },
            )
            llm_json = (
                '{"action":"append","sections":[{"title":"Rule","content":"## Rule\\n\\nUse it.",'
                '"covered_case_ids":["output-primary"],"supporting_case_ids":["output-ref"],'
                '"reason":"general"}],"reason":"ok"}'
            )
            with patch("skill_compact._llm_call", return_value=llm_json):
                result = promote_effective_cases(object(), "model", skills_dir=skills_dir, verbose=False)
            self.assertEqual(result[0]["compressed_case_ids"], ["output-primary", "output-ref"])
            changed = compact_compressed_case_bodies(skills_dir, archive_dir=archive_dir)
            self.assertEqual(changed, ["skill_output_cases.md"])
            active = (skills_dir / "skill_output_cases.md").read_text(encoding="utf-8")
            archived = (archive_dir / "skill_output_cases_archive.md").read_text(encoding="utf-8")
            self.assertNotIn("output-primary", active)
            self.assertNotIn("output-ref", active)
            self.assertIn("primary body", archived)
            self.assertIn("reference body", archived)

    def test_case_dedup_stops_after_strict_pass_and_only_touches_requested_side(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills_dir = root / "skills"
            archive_dir = root / "archive"
            for idx in range(3):
                append_skill_case(
                    skills_dir,
                    side="input",
                    heading=f"## Input case {idx}",
                    body=f"**When:** repeated pattern {idx}\n\n**Fix:** {'x' * 600}",
                    meta={
                        "case_id": f"input-{idx}",
                        "side": "input",
                        "usage": {"compressed": False},
                    },
                )
            append_skill_case(
                skills_dir,
                side="output",
                heading="## Output case",
                body="**When:** unrelated\n\n**Fix:** keep",
                meta={
                    "case_id": "output-keep",
                    "side": "output",
                    "usage": {"compressed": True},
                },
            )
            case_path = skills_dir / "skill_input_cases.md"
            output_before = (skills_dir / "skill_output_cases.md").read_text(encoding="utf-8")
            initial_chars = len(case_path.read_text(encoding="utf-8"))
            threshold = int(initial_chars * 0.8)

            with patch(
                "skill_compact._llm_call",
                return_value='{"archive_case_ids":["input-0"],"reason":"duplicate"}',
            ) as llm_call:
                result = deduplicate_skill_file(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    side="input",
                    kind="cases",
                    threshold_chars=threshold,
                    archive_dir=archive_dir,
                    checkpoint_name="ckpt_130",
                    verbose=False,
                )

            self.assertEqual(llm_call.call_count, 1)
            self.assertTrue(result["triggered"])
            self.assertTrue(result["threshold_met"])
            self.assertEqual(result["stages"][0]["stage"], "strict")
            self.assertEqual({case.case_id for case in iter_skill_cases(skills_dir, side="input")}, {"input-1", "input-2"})
            self.assertEqual(
                (skills_dir / "skill_output_cases.md").read_text(encoding="utf-8"),
                output_before,
            )
            self.assertIn(
                "input-0",
                (archive_dir / "skill_input_cases_archive.md").read_text(encoding="utf-8"),
            )

    def test_case_dedup_uses_relaxed_pass_to_reach_85_percent_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills_dir = root / "skills"
            archive_dir = root / "archive"
            for idx in range(4):
                append_skill_case(
                    skills_dir,
                    side="chain",
                    heading=f"## Chain case {idx}",
                    body=f"**When:** similar chain {idx}\n\n**Fix:** {'y' * 700}",
                    meta={
                        "case_id": f"chain-{idx}",
                        "side": "chain",
                        "usage": {"compressed": False},
                    },
                )
            case_path = skills_dir / "skill_chain_cases.md"
            initial_chars = len(case_path.read_text(encoding="utf-8"))
            threshold = int(initial_chars * 0.55)
            target = int(threshold * 0.85)

            with patch(
                "skill_compact._llm_call",
                side_effect=[
                    '{"archive_case_ids":["chain-0"],"reason":"clear duplicate"}',
                    '{"archive_case_ids":["chain-1","chain-2"],"reason":"broader duplicate group"}',
                ],
            ) as llm_call:
                result = deduplicate_skill_file(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    side="chain",
                    kind="cases",
                    threshold_chars=threshold,
                    target_ratio=0.85,
                    archive_dir=archive_dir,
                    checkpoint_name="ckpt_140",
                    verbose=False,
                )

            self.assertEqual(llm_call.call_count, 2)
            self.assertEqual([stage["stage"] for stage in result["stages"]], ["strict", "relaxed"])
            self.assertTrue(result["target_met"])
            self.assertLessEqual(result["final_chars"], target)
            self.assertEqual([case.case_id for case in iter_skill_cases(skills_dir, side="chain")], ["chain-3"])

    def test_case_dedup_reprompts_when_llm_selects_every_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills_dir = root / "skills"
            archive_dir = root / "archive"
            for idx in range(3):
                append_skill_case(
                    skills_dir,
                    side="chain",
                    heading=f"## Chain case {idx}",
                    body=f"**When:** similar chain {idx}\n\n**Fix:** {'z' * 600}",
                    meta={
                        "case_id": f"chain-{idx}",
                        "side": "chain",
                        "usage": {"compressed": False},
                    },
                )
            case_path = skills_dir / "skill_chain_cases.md"
            initial_chars = len(case_path.read_text(encoding="utf-8"))
            threshold = int(initial_chars * 0.8)

            with patch(
                "skill_compact._llm_call",
                side_effect=[
                    json.dumps({
                        "archive_case_ids": ["chain-0", "chain-1", "chain-2"],
                        "reason": "all look similar",
                    }),
                    json.dumps({
                        "archive_case_ids": ["chain-0"],
                        "reason": "keep chain-1 and chain-2 as representatives",
                    }),
                ],
            ) as llm_call:
                result = deduplicate_skill_file(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    side="chain",
                    kind="cases",
                    threshold_chars=threshold,
                    archive_dir=archive_dir,
                    checkpoint_name="ckpt_140",
                    verbose=False,
                )

            self.assertEqual(llm_call.call_count, 2)
            self.assertEqual(result["stages"][0]["status"], "applied")
            self.assertEqual(
                result["stages"][0]["correction_reason"],
                "initial_response_selected_all_cases",
            )
            self.assertEqual(result["stages"][0]["invalid_attempts"][0]["status"], "rejected_all_cases")
            self.assertEqual(
                {case.case_id for case in iter_skill_cases(skills_dir, side="chain")},
                {"chain-1", "chain-2"},
            )

    def test_general_dedup_archives_original_and_stops_after_strict_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills_dir = root / "skills"
            archive_dir = root / "archive"
            skill_path = skills_dir / "skill_output.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            original = "## Rule A\n\n" + ("same guidance\n" * 120)
            compacted = "## Rule A\n\nUse the shared guidance once.\n"
            skill_path.write_text(original, encoding="utf-8")

            with patch(
                "skill_compact._llm_call",
                return_value=json.dumps({"content": compacted, "reason": "merged duplicates"}),
            ) as llm_call:
                result = deduplicate_skill_file(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    side="output",
                    kind="general",
                    threshold_chars=500,
                    archive_dir=archive_dir,
                    checkpoint_name="ckpt_150",
                    verbose=False,
                )

            self.assertEqual(llm_call.call_count, 1)
            self.assertTrue(result["threshold_met"])
            self.assertEqual(skill_path.read_text(encoding="utf-8"), compacted)
            archives = list((archive_dir / "general_skills").glob("*.md"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(encoding="utf-8"), original)

    def test_general_dedup_relaxed_pass_must_reach_85_percent_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skills_dir = root / "skills"
            archive_dir = root / "archive"
            skill_path = skills_dir / "skill_input.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            original = "## Rule\n\n" + ("guidance\n" * 200)
            strict_content = "## Rule\n\n" + ("guidance\n" * 90)
            relaxed_content = "## Rule\n\n" + ("guidance\n" * 35)
            skill_path.write_text(original, encoding="utf-8")
            threshold = 600
            target = int(threshold * 0.85)

            with patch(
                "skill_compact._llm_call",
                side_effect=[
                    json.dumps({"content": strict_content, "reason": "strict merge"}),
                    json.dumps({"content": relaxed_content, "reason": "relaxed merge"}),
                ],
            ) as llm_call:
                result = deduplicate_skill_file(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    side="input",
                    kind="general",
                    threshold_chars=threshold,
                    target_ratio=0.85,
                    archive_dir=archive_dir,
                    checkpoint_name="ckpt_150",
                    verbose=False,
                )

            self.assertEqual(llm_call.call_count, 2)
            self.assertEqual([stage["stage"] for stage in result["stages"]], ["strict", "relaxed"])
            self.assertTrue(result["target_met"])
            self.assertLessEqual(result["final_chars"], target)
            self.assertEqual(skill_path.read_text(encoding="utf-8"), relaxed_content)


    def test_agent_case_write_adds_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td)
            (skills_dir / "skill_output_cases.md").write_text("", encoding="utf-8")
            skills_token = agent_proposer._CURRENT_SKILLS_DIR.set(skills_dir)
            learning_token = agent_proposer._SKILL_LEARNING_ENABLED.set(True)
            context_token = agent_proposer._CURRENT_FEATURE_CONTEXT.set({
                "layer_id": "6",
                "feature_id": "600",
                "round_idx": 3,
            })
            events_token = agent_proposer._CURRENT_TOOL_EVENTS.set([])
            try:
                result = agent_proposer._run_write_skill(
                    "skill_output_cases.md",
                    "## Agent case\n\n**When:** demo\n\n**Fix:** demo",
                    "append",
                )
                self.assertIn("OK: appended case", result)
                cases = iter_skill_cases(skills_dir, side="output")
                self.assertEqual(len(cases), 1)
                self.assertEqual(cases[0].meta["write_trigger"], "agent_decision")
                events = agent_proposer._CURRENT_TOOL_EVENTS.get()
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["event"], "case_write")
                self.assertEqual(events[0]["result"], "ok")
                self.assertEqual(events[0]["kind"], "case")
                self.assertEqual(events[0]["case_id"], cases[0].case_id)
                self.assertEqual(events[0]["feature_id"], "600")
            finally:
                agent_proposer._CURRENT_TOOL_EVENTS.reset(events_token)
                agent_proposer._CURRENT_FEATURE_CONTEXT.reset(context_token)
                agent_proposer._SKILL_LEARNING_ENABLED.reset(learning_token)
                agent_proposer._CURRENT_SKILLS_DIR.reset(skills_token)

    def test_skill_read_file_records_tool_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td)
            (skills_dir / "skill_output_cases.md").write_text("## Case\n\nbody\n", encoding="utf-8")
            skills_token = agent_proposer._CURRENT_SKILLS_DIR.set(skills_dir)
            context_token = agent_proposer._CURRENT_FEATURE_CONTEXT.set({
                "layer_id": "1",
                "feature_id": "2",
                "current_timestamp": "ts",
                "round_idx": 4,
            })
            events_token = agent_proposer._CURRENT_TOOL_EVENTS.set([])
            try:
                result = agent_proposer._run_read_file("skills/skill_output_cases.md")
                self.assertIn("## Case", result)
                events = agent_proposer._CURRENT_TOOL_EVENTS.get()
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["event"], "skill_read_file")
                self.assertEqual(events[0]["path"], "skills/skill_output_cases.md")
                self.assertEqual(events[0]["filename"], "skill_output_cases.md")
                self.assertEqual(events[0]["kind"], "case")
                self.assertEqual(events[0]["feature_id"], "2")
                self.assertFalse(events[0]["truncated"])
            finally:
                agent_proposer._CURRENT_TOOL_EVENTS.reset(events_token)
                agent_proposer._CURRENT_FEATURE_CONTEXT.reset(context_token)
                agent_proposer._CURRENT_SKILLS_DIR.reset(skills_token)

    def test_log_tools_are_restricted_to_current_feature_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            logs_root = Path(td) / "logs"
            current_dir = logs_root / "layer-6" / "feature-600" / "ts"
            other_dir = logs_root / "layer-6" / "feature-601" / "ts"
            current_dir.mkdir(parents=True)
            other_dir.mkdir(parents=True)
            (current_dir / "trace.json").write_text('{"feature": 600}\n', encoding="utf-8")
            (other_dir / "trace.json").write_text('{"feature": 601}\n', encoding="utf-8")

            logs_token = agent_proposer._CURRENT_LOGS_ROOT.set(logs_root)
            context_token = agent_proposer._CURRENT_FEATURE_CONTEXT.set({
                "layer_id": "6",
                "feature_id": "600",
                "current_timestamp": "ts",
                "round_idx": 1,
            })
            try:
                current = agent_proposer._run_read_file(
                    "logs/layer-6/feature-600/ts/trace.json"
                )
                other = agent_proposer._run_read_file(
                    "logs/layer-6/feature-601/ts/trace.json"
                )
                absolute_other = agent_proposer._run_read_file(str(other_dir / "trace.json"))
                escaped = agent_proposer._run_read_file(
                    "logs/layer-6/feature-600/../feature-601/ts/trace.json"
                )
                feature_root_listing = agent_proposer._run_list_directory("logs")
                current_listing = agent_proposer._run_list_directory(
                    "logs/layer-6/feature-600/ts"
                )
                other_listing = agent_proposer._run_list_directory(
                    "logs/layer-6/feature-601/ts"
                )

                self.assertIn('"feature": 600', current)
                self.assertIn("ERROR", other)
                self.assertIn("ERROR", absolute_other)
                self.assertIn("ERROR", escaped)
                self.assertIn("ts", feature_root_listing)
                self.assertNotIn("feature-601", feature_root_listing)
                self.assertIn("trace.json", current_listing)
                self.assertIn("ERROR", other_listing)
            finally:
                agent_proposer._CURRENT_FEATURE_CONTEXT.reset(context_token)
                agent_proposer._CURRENT_LOGS_ROOT.reset(logs_token)


    def test_skill_archive_files_are_hidden_from_agent_tools(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td)
            (skills_dir / "skill_output_cases.md").write_text("## visible\n", encoding="utf-8")
            (skills_dir / "skill_output_cases_archive.md").write_text("## hidden\n", encoding="utf-8")
            skills_token = agent_proposer._CURRENT_SKILLS_DIR.set(skills_dir)
            events_token = agent_proposer._CURRENT_TOOL_EVENTS.set([])
            try:
                listing = agent_proposer._run_list_directory("skills")
                self.assertIn("skill_output_cases.md", listing)
                self.assertNotIn("skill_output_cases_archive.md", listing)
                read_result = agent_proposer._run_read_file("skills/skill_output_cases_archive.md")
                self.assertIn("ERROR", read_result)
                self.assertEqual(agent_proposer._CURRENT_TOOL_EVENTS.get(), [])
            finally:
                agent_proposer._CURRENT_TOOL_EVENTS.reset(events_token)
                agent_proposer._CURRENT_SKILLS_DIR.reset(skills_token)


    def test_skill_learning_disabled_hides_write_tool(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_token = agent_proposer._CURRENT_SKILLS_DIR.set(Path(td))
            learning_token = agent_proposer._SKILL_LEARNING_ENABLED.set(False)
            events_token = agent_proposer._CURRENT_TOOL_EVENTS.set([])
            try:
                tool_names = {tool["function"]["name"] for tool in agent_proposer._tools_for_current_context()}
                self.assertNotIn("write_skill", tool_names)
                result = agent_proposer._run_write_skill("skill_input.md", "## nope", "create")
                self.assertIn("disabled", result)
                self.assertFalse((Path(td) / "skill_input.md").exists())
                events = agent_proposer._CURRENT_TOOL_EVENTS.get()
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["event"], "skill_write")
                self.assertEqual(events[0]["result"], "error")
            finally:
                agent_proposer._CURRENT_TOOL_EVENTS.reset(events_token)
                agent_proposer._SKILL_LEARNING_ENABLED.reset(learning_token)
                agent_proposer._CURRENT_SKILLS_DIR.reset(skills_token)


if __name__ == "__main__":
    unittest.main()
