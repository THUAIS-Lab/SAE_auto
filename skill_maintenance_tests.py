from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_proposer
from skill_case_store import append_skill_case, increment_case_usage
from skill_lock import exclusive_skill_lock, shared_skill_lock
from skill_maintenance import (
    CASE_DEDUP_THRESHOLDS,
    _hash_tree,
    maybe_run_auto_skill_maintenance,
    oversized_skill_files,
)
from self_evolution_utils import hash_tree


class SkillMaintenanceTest(unittest.TestCase):
    def test_under_threshold_skips_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td) / "skills"
            skills_dir.mkdir()
            (skills_dir / "skill_input_cases.md").write_text("small\n", encoding="utf-8")

            with patch("skill_maintenance._perform_skill_maintenance") as perform:
                result = maybe_run_auto_skill_maintenance(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    archive_dir=Path(td) / "archive",
                    maintenance_name="test",
                    verbose=False,
                )

            self.assertEqual(result["status"], "not_needed")
            perform.assert_not_called()

    def test_over_threshold_runs_maintenance_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td) / "skills"
            skills_dir.mkdir()
            (skills_dir / "skill_input_cases.md").write_text(
                "x" * (CASE_DEDUP_THRESHOLDS["input"] + 1),
                encoding="utf-8",
            )
            expected = {"status": "ok", "maintenance_name": "test"}
            with patch("skill_maintenance._perform_skill_maintenance", return_value=dict(expected)) as perform:
                result = maybe_run_auto_skill_maintenance(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    archive_dir=Path(td) / "archive",
                    maintenance_name="test",
                    verbose=False,
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["triggered_by"], "post_write_threshold")
            perform.assert_called_once()

    def test_threshold_is_rechecked_after_waiting_for_maintenance_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td) / "skills"
            skills_dir.mkdir()
            oversized = [{"side": "input", "kind": "cases", "file": "skill_input_cases.md"}]
            with patch(
                "skill_maintenance.oversized_skill_files",
                side_effect=[oversized, []],
            ), patch("skill_maintenance._perform_skill_maintenance") as perform:
                result = maybe_run_auto_skill_maintenance(
                    object(),
                    "model",
                    skills_dir=skills_dir,
                    archive_dir=Path(td) / "archive",
                    maintenance_name="test",
                    verbose=False,
                )
            self.assertEqual(result["status"], "not_needed")
            self.assertEqual(result["initial_oversized_files"], oversized)
            perform.assert_not_called()

    def test_skill_lock_is_reentrant_but_does_not_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td) / "skills"
            with shared_skill_lock(skills_dir):
                with shared_skill_lock(skills_dir):
                    pass
                with self.assertRaisesRegex(RuntimeError, "cannot upgrade"):
                    with exclusive_skill_lock(skills_dir):
                        pass
            with exclusive_skill_lock(skills_dir):
                with shared_skill_lock(skills_dir):
                    pass

    def test_case_post_write_runs_only_for_a_new_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td) / "skills"
            calls = []
            kwargs = {
                "side": "input",
                "heading": "## [Case][input] example",
                "body": "body",
                "meta": {"case_id": "input-test-case"},
                "post_write": lambda: calls.append("called"),
            }
            append_skill_case(skills_dir, **kwargs)
            append_skill_case(skills_dir, **kwargs)
            self.assertEqual(calls, ["called"])

    def test_case_usage_update_triggers_post_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td) / "skills"
            append_skill_case(
                skills_dir,
                side="input",
                heading="## [Case][input] example",
                body="body",
                meta={"case_id": "input-usage-case"},
            )
            calls = []
            increment_case_usage(
                skills_dir,
                "input-usage-case",
                use_delta=1,
                post_write=lambda: calls.append("called"),
            )
            self.assertEqual(calls, ["called"])

    def test_agent_general_skill_append_triggers_auto_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td) / "skills"
            skills_dir.mkdir()
            (skills_dir / "skill_input.md").write_text("existing\n", encoding="utf-8")
            skills_token = agent_proposer._CURRENT_SKILLS_DIR.set(skills_dir)
            maintenance_token = agent_proposer._CURRENT_SKILL_MAINTENANCE.set(
                {
                    "mode": "auto",
                    "client": object(),
                    "model": "model",
                    "archive_dir": Path(td) / "archive",
                    "maintenance_name": "test",
                }
            )
            try:
                with patch(
                    "agent_proposer.maybe_run_auto_skill_maintenance",
                    return_value={"status": "not_needed"},
                ) as maintenance:
                    result = agent_proposer._run_write_skill("skill_input.md", "new section", "append")
            finally:
                agent_proposer._CURRENT_SKILL_MAINTENANCE.reset(maintenance_token)
                agent_proposer._CURRENT_SKILLS_DIR.reset(skills_token)

            self.assertTrue(result.startswith("OK:"))
            maintenance.assert_called_once()

    def test_oversized_files_are_independently_thresholded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td)
            (skills_dir / "skill_chain_cases.md").write_text(
                "x" * (CASE_DEDUP_THRESHOLDS["chain"] + 1),
                encoding="utf-8",
            )
            result = oversized_skill_files(skills_dir)
            self.assertEqual([(item["side"], item["kind"]) for item in result], [("chain", "cases")])

    def test_maintenance_hash_ignores_runtime_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            skills_dir = Path(td)
            (skills_dir / "skill_input.md").write_text("content\n", encoding="utf-8")
            before = _hash_tree(skills_dir)
            self.assertEqual(before, hash_tree(skills_dir))
            (skills_dir / ".maintenance.lock").write_text("runtime\n", encoding="utf-8")
            self.assertEqual(_hash_tree(skills_dir), before)


if __name__ == "__main__":
    unittest.main()
