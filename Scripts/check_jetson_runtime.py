from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str



def _module_version(module_name: str) -> CheckResult:
    #checking the module (or library version) of a specific library
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(module_name, False, f"import failed: {type(exc).__name__}: {exc}")

    version = getattr(module, "__version__", "version unavailable")
    module_file = getattr(module, "__file__", "built-in")
    return CheckResult(module_name, True, f"{version} ({module_file})")

def _cuda_runtime() -> CheckResult:
    errors = []
    for module_name in ("cuda.bindings.runtime", "cuda.cudart"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue

        module_file = getattr(module,"__file__", "built-in")
        return CheckResult("CUDA Python runtime", True, f"{module_name} ({module_file})")

    #if there is no cuda
    return CheckResult("CUDA Python runtime", False, "; ".join(errors))
    

def _venv_uses_system_packages() -> CheckResult:
    if sys.prefix == sys.base_prefix:
        return CheckResult(
            "virtual environment",
            False,
            "not running inside a virtual environment",
        )

    config_path = Path(sys.prefix) / "pyvenv.cfg"
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult("system site-packages", False, f"cannot read {config_path}: {exc}")

    enabled = any(
        line.strip().lower() == "include-system-site-packages = true"
        for line in config_text.splitlines()
    )
    if not enabled:
        return CheckResult(
            "system site-packages",
            False,
            f"{config_path} does not enable include-system-site-packages",
        )
    return CheckResult("system site-packages", True, str(config_path))


def _executable(name: str, *, required: bool) -> CheckResult:
    resolved = shutil.which(name)
    if resolved is not None:
        return CheckResult(name, True, resolved)
    requirement = "required" if required else "optional"
    return CheckResult(name, not required, f"not found on PATH ({requirement})")


def collect_results(*, require_ffmpeg: bool = False) -> Sequence[CheckResult]:
    python_ok = sys.version_info[:2] == (3, 10)
    results = [
        CheckResult(
            "Python",
            python_ok,
            f"{sys.version.split()[0]} ({sys.executable}); required: 3.10.x",
        ),
        _venv_uses_system_packages(),
        _module_version("numpy"),
        _module_version("cv2"),
        _module_version("tensorrt"),
        _cuda_runtime(),
        _executable("trtexec", required=True),
        _executable("ffmpeg", required=require_ffmpeg),
    ]
    return tuple(results)


def _print_results(results: Sequence[CheckResult]) -> None:
    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a Jetson virtual environment can use the Python 3.10, "
            "TensorRT, CUDA, NumPy, and OpenCV components required by the SOP runner."
        )
    )
    parser.add_argument(
        "--require-ffmpeg",
        action="store_true",
        help="Fail when ffmpeg is unavailable instead of reporting it as optional.",
    )
    args = parser.parse_args(argv)

    results = collect_results(require_ffmpeg=bool(args.require_ffmpeg))
    _print_results(results)
    failures = [result for result in results if not result.ok]
    if failures:
        print("\nJetson runtime is not ready. Fix the failed checks before building or running an engine.")
        return 1

    print("\nJetson runtime prerequisites are visible inside the virtual environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
