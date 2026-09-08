"""
Docstring for Scripts.build_trt_engine
Function : For creating TRT Engine for edge deployment
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class TrtBuildSpec:
    onnx: Path
    engine: Path
    input_name: str
    imgsz: int
    fp16: bool
    int8: bool
    workspace_mb: Optional[int]
    timing_cache: Optional[Path]
    verbose: bool
    dry_run: bool


def _maybe_ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_repo_path(raw_path: str, repo_root: Path) -> Path:
    """Resolve paths from either the current directory or this repository."""
    path = Path(raw_path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    repo_path = repo_root / path
    if repo_path.exists() or not path.parent.exists():
        return repo_path
    return path


def _find_trtexec() -> str:
    candidates = [
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(
        "Could not find trtexec. Activate the Jetson environment or add "
        "/usr/src/tensorrt/bin to PATH."
    )


def _trtexec_help(executable: str) -> str:
    result = subprocess.run(
        [executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return f"{result.stdout}\n{result.stderr}"


def _compatible_trtexec_options(help_text: str, spec: TrtBuildSpec) -> List[str]:
    """Select flags shared by TensorRT generations used on Jetson."""
    options: List[str] = []

    # TensorRT 10 uses --skipInference; older releases may expose --buildOnly.
    if "--skipInference" in help_text:
        options.append("--skipInference")
    elif "--buildOnly" in help_text:
        options.append("--buildOnly")

    if spec.workspace_mb is not None:
        if "--memPoolSize" in help_text:
            options.append(f"--memPoolSize=workspace:{int(spec.workspace_mb)}")
        elif "--workspace" in help_text:
            options.append(f"--workspace={int(spec.workspace_mb)}")
        else:
            print(
                "WARNING: this trtexec has no recognized workspace flag; "
                "continuing with its default workspace.",
                file=sys.stderr,
            )

    return options


def _build_trtexec_cmd(
    spec: TrtBuildSpec,
    *,
    executable: str = "trtexec",
    optional_options: Sequence[str] = (),
) -> List[str]:
    cmd = [
        executable,
        f"--onnx={str(spec.onnx)}",
        f"--saveEngine={str(spec.engine)}",
    ]
    cmd.extend(optional_options)

    if spec.verbose:
        cmd.append("--verbose")

    if spec.fp16:
        cmd.append("--fp16")

    if spec.int8:
        cmd.append("--int8")

    if spec.timing_cache is not None:
        _maybe_ensure_parent(spec.timing_cache)
        cmd.append(f"--timingCacheFile={str(spec.timing_cache)}")

    # For YOLO-style exports, input is usually NCHW with fixed shape.
    shape = f"1x3x{int(spec.imgsz)}x{int(spec.imgsz)}"
    cmd.append(f"--shapes={spec.input_name}:{shape}")

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TensorRT engine (.engine) from an ONNX model using `trtexec`.\n\n"
            "Important:\n"
            "- Build the engine on the target device (Jetson) because TRT engines are hardware-specific.\n"
            "- This script does not require Python TensorRT; it shells out to `trtexec`."
        )
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX Model (contohnya Models/action_model.onnx).")
    parser.add_argument(
        "--engine",
        default=None,
        help="Output engine path. Default: same as --onnx but with .engine extension.",
    )
    parser.add_argument(
        "--input-name",
        default="images",
        help='ONNX input tensor name (common: "images").'
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Static input size (e.g., 640, 960).")
    parser.add_argument("--fp16", action="store_true", help="Build FP16 engine (recommended on Jetson).")
    parser.add_argument("--int8", action="store_true", help="Build INT8 engine (requires calibratable model/setup).")
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=None,
        help="TensorRT workspace size in MB (optional).",
    )
    parser.add_argument(
        "--timing-cache",
        default=None,
        help="Optional timing cache file (speeds up rebuilds).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable trtexec verbose logs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the trtexec command but do not execute.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    onnx_path = _resolve_repo_path(str(args.onnx), repo_root)
    if not onnx_path.exists():
        raise FileNotFoundError(str(onnx_path))
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError("--onnx must point to a .onnx file")

    if args.imgsz < 32:
        raise ValueError("--imgsz must be >= 32")
    if args.workspace_mb is not None and args.workspace_mb < 1:
        raise ValueError("--workspace-mb bust be >= 1")
    if args.int8 and not args.fp16:
        # Not strictly required, but a commaon expectation on jetson builds."
        print("WARNING: buidling INT8 without --fp16; ensure this is intentional.")

    engine_path = (
        _resolve_repo_path(str(args.engine), repo_root)
        if args.engine
        else onnx_path.with_suffix(".engine")
    )
    _maybe_ensure_parent(engine_path)

    timing_cache = (
        _resolve_repo_path(str(args.timing_cache), repo_root)
        if args.timing_cache
        else None
    )
    spec = TrtBuildSpec(
        onnx=onnx_path,
        engine=engine_path,
        input_name=str(args.input_name),
        imgsz=int(args.imgsz),
        fp16=bool(args.fp16),
        int8=bool(args.int8),
        workspace_mb=args.workspace_mb,
        timing_cache=timing_cache,
        verbose=bool(args.verbose),
        dry_run=bool(args.dry_run),
    )

    try:
        executable = _find_trtexec()
    except FileNotFoundError:
        if not spec.dry_run:
            raise
        executable = "trtexec"

    help_text = _trtexec_help(executable) if executable != "trtexec" or not spec.dry_run else ""
    optional_options = _compatible_trtexec_options(help_text, spec)
    cmd = _build_trtexec_cmd(
        spec,
        executable=executable,
        optional_options=optional_options,
    )
    print("CMD:", " ".join(shlex.quote(c) for c in cmd))

    if spec.dry_run:
        return 0

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed with exit code {proc.returncode}")

    if not engine_path.exists():
        raise RuntimeError(f"trtexec reported success but engine not found: {engine_path}")

    print(f"wrote {engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
