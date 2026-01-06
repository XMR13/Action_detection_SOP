"""
Docstring for Scripts.build_trt_engine
Function : For creating TRT Engine for edge deployment
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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


def _build_trtexec_cmd(spec: TrtBuildSpec) -> List[str]:
    cmd = [
        "trtexec",
        f"--onnx={str(spec.onnx)}",
        f"--saveEngine={str(spec.engine)}",
        "--buildOnly",
    ]

    if spec.verbose:
        cmd.append("--verbose")

    if spec.fp16:
        cmd.append("--fp16")

    if spec.int8:
        cmd.append("--int8")

    if spec.workspace_mb is not None:
        # NOTE: TensorRT flag name varies across versions; `--workspace` is widely supported.
        cmd.append(f"--workspace={int(spec.workspace_mb)}")

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

    #process the args
    onnx_path = Path(args.onnx)
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

    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix(".engine")
    _maybe_ensure_parent(engine_path)

    timing_cache = Path(args.timing_cache) if args.timing_cache else None
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

    cmd = _build_trtexec_cmd(spec)
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

