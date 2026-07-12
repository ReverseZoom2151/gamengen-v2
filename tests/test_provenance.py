from src.utils.training import build_run_manifest


def test_run_manifest_hashes_config_and_existing_data_metadata(tmp_path):
    (tmp_path / "metadata.json").write_text('{"schema_version":2}')
    manifest = build_run_manifest({"seed": 4}, str(tmp_path))
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["data_metadata_sha256"]) == 64
    assert "torch" in manifest
