from pathlib import Path

from Scripts.build_trt_engine import (
    TrtBuildSpec,
    _build_trtexec_cmd,
    _compatible_trtexec_options,
    _resolve_repo_path,
)


def _spec(*, workspace_mb: int | None = None) -> TrtBuildSpec:
    return TrtBuildSpec(
        onnx=Path("model.onnx"),
        engine=Path("model.engine"),
        input_name="images",
        imgsz=640,
        fp16=True,
        int8=False,
        workspace_mb=workspace_mb,
        timing_cache=None,
        verbose=False,
        dry_run=True,
    )


def test_trt10_options_use_skip_inference_and_mem_pool() -> None:
    options = _compatible_trtexec_options(
        "--skipInference\n--memPoolSize=<pool_spec>",
        _spec(workspace_mb=1024),
    )

    assert options == ["--skipInference", "--memPoolSize=workspace:1024"]


def test_older_trt_options_use_build_only_and_workspace() -> None:
    options = _compatible_trtexec_options(
        "--buildOnly\n--workspace=N",
        _spec(workspace_mb=512),
    )

    assert options == ["--buildOnly", "--workspace=512"]


def test_unknown_trt_build_flags_still_produce_a_build_command() -> None:
    cmd = _build_trtexec_cmd(
        _spec(),
        executable="/opt/tensorrt/trtexec",
        optional_options=(),
    )

    assert cmd == [
        "/opt/tensorrt/trtexec",
        "--onnx=model.onnx",
        "--saveEngine=model.engine",
        "--fp16",
        "--shapes=images:1x3x640x640",
    ]


def test_relative_paths_fall_back_to_repository_when_called_elsewhere(tmp_path: Path) -> None:
    (tmp_path / "Models").mkdir()
    model = tmp_path / "Models" / "model.onnx"
    model.write_bytes(b"onnx")

    assert _resolve_repo_path("Models/model.onnx", tmp_path) == model
