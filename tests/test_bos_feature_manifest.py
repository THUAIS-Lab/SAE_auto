from __future__ import annotations

import sys
from pathlib import Path

import run_main_experiment
from prepare_bos_feature_manifest import (
    discover_sage_features,
    load_feature_manifest,
    validate_feature_map,
    write_feature_manifest,
)


def test_manifest_discovers_layer_feature_directories(tmp_path: Path) -> None:
    for layer_id, feature_ids in {0: [30, 60], 6: [90, 120]}.items():
        for feature_id in feature_ids:
            (tmp_path / f"layer_{layer_id}" / f"feature_{feature_id}").mkdir(parents=True)
    (tmp_path / "layer_0" / "README.txt").write_text("ignored", encoding="utf-8")

    layers = discover_sage_features(tmp_path)
    validate_feature_map(layers, expected_layers=[0, 6], expected_total=4, expected_per_layer=2)
    output = tmp_path / "manifest.json"
    payload = write_feature_manifest(tmp_path, output, layers)

    assert payload["total_features"] == 4
    assert load_feature_manifest(output) == {0: [30, 60], 6: [90, 120]}
    assert (tmp_path / "manifest.layer-0.txt").read_text(encoding="utf-8") == "30\n60\n"


def test_manifest_validation_rejects_wrong_count() -> None:
    try:
        validate_feature_map({0: [30]}, expected_layers=[0], expected_total=2)
    except ValueError as exc:
        assert "expected 2 total features" in str(exc)
    else:
        raise AssertionError("expected total mismatch")


def test_main_dry_run_uses_manifest_layers_and_features(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manifest = tmp_path / "manifest.json"
    write_feature_manifest(tmp_path, manifest, {0: [30, 60], 6: [90, 120]})
    sae_paths = {0: tmp_path / "sae-0", 6: tmp_path / "sae-6"}
    for sae_path in sae_paths.values():
        sae_path.mkdir()
    monkeypatch.setattr(run_main_experiment, "CODE_DIR", tmp_path)
    monkeypatch.setattr(
        run_main_experiment,
        "SAE_PATHS_16K",
        {layer: str(path) for layer, path in sae_paths.items()},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_main_experiment.py",
            "--timestamp",
            "manifest-dry-run",
            "--logs-root",
            str(tmp_path / "logs"),
            "--feature-manifest",
            str(manifest),
            "--dry-run",
        ],
    )

    run_main_experiment.main()

    output = capsys.readouterr().out
    assert "layers=[0, 6]" in output
    assert "layer 0: 30 60" in output
    assert "layer 6: 90 120" in output


def test_bos_preflight_rejects_wrong_feature_metadata(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "layer-6" / "feature-900" / "bos_token" / "prompt-0001"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "top_tokens.json").write_text(
        '{"layer_id": 6, "feature_id": 930, "prompt_id": "prompt-0001"}',
        encoding="utf-8",
    )

    try:
        run_main_experiment._validate_bos_prompt_artifacts(
            bos_token_root=tmp_path,
            prompt_id="prompt-0001",
            layer_feature_ids={6: [900]},
        )
    except ValueError as exc:
        assert "metadata mismatch" in str(exc)
    else:
        raise AssertionError("preflight must reject another feature's observation")
