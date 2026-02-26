from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from Action_Detection_SOP.reporting import today_date_str
from Action_Detection_SOP.run_config import apply_run_config, collect_cli_dests
from Action_Detection_SOP.runner_mvp import run_mvp
from Scripts.run_sop_mvp import build_parser

ALLOWED_EXPECT_KEYS = {
    "total_sessions",
    "roi_done",
    "roi_not_done",
    "roi_unknown",
    "helmet_done",
    "helmet_not_done",
    "helmet_unknown",
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    out_dir: str
    passed: bool
    expected: Dict[str, int]
    actual: Dict[str, int]
    errors: List[str]


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set manifest not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid golden set manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Golden set manifest must be a JSON object")
    return payload


def _merge_run_args(base: Dict[str, object], overrides: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base)
    merged.update(overrides)
    if "source" in overrides:
        merged["source"] = overrides["source"]
    return merged


def _load_daily_report(out_dir: Path) -> Dict[str, int]:
    date = today_date_str()
    report_path = out_dir / "reports" / date / "daily_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Expected daily report missing: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"daily_report.json must be an object: {report_path}")
    return {k: int(v) for k, v in payload.items() if k in ALLOWED_EXPECT_KEYS}


def _compare_expectations(expected: Dict[str, int], actual: Dict[str, int]) -> List[str]:
    errors: List[str] = []
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        if act_val is None:
            errors.append(f"Missing actual key: {key}")
            continue
        if act_val != exp_val:
            errors.append(f"{key}: expected {exp_val}, got {act_val}")
    return errors


def _validate_expect(expect: object) -> Dict[str, int]:
    if expect is None:
        return {}
    if not isinstance(expect, dict):
        raise ValueError("expect must be an object")
    unknown = sorted(k for k in expect.keys() if k not in ALLOWED_EXPECT_KEYS)
    if unknown:
        raise ValueError(f"Unknown expect keys: {unknown}")
    clean: Dict[str, int] = {}
    for key, value in expect.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"expect.{key} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"expect.{key} must be an integer")
        clean[key] = int(value)
    return clean


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden set regression runner for SOP MVP-A.")
    ap.add_argument("--manifest", required=True, help="Path to golden set manifest JSON.")
    ap.add_argument("--out-dir", default="data/golden_set", help="Base output directory for golden set runs.")
    ap.add_argument("--stop-on-fail", action="store_true", help="Stop at first failing case.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    payload = _load_manifest(manifest_path)
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError("golden set schema_version must be 1")

    base = payload.get("base", {})
    if base is None:
        base = {}
    if not isinstance(base, dict):
        raise ValueError("manifest.base must be an object")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.cases must be a non-empty list")

    parser = build_parser()
    cli_dests = collect_cli_dests(parser, [])
    results: List[CaseResult] = []

    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case[{idx}] must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case[{idx}].id must be a non-empty string")

        overrides = case.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"case[{idx}].overrides must be an object")

        run_args = _merge_run_args(base, overrides)
        if "out_dir" not in run_args:
            run_args["out_dir"] = str(Path(args.out_dir) / case_id)
        if "show" not in run_args:
            run_args["show"] = False
        if "progress" not in run_args:
            run_args["progress"] = False

        if "source" not in run_args and not any(k in run_args for k in ("video", "webcam", "rtsp")):
            raise ValueError(f"case[{idx}] must define a source (video/webcam/rtsp)")

        args_ns = parser.parse_args([])
        apply_run_config(args=args_ns, payload=run_args, cli_dests=cli_dests, parser=parser)
        args_raw = dict(vars(args_ns))

        print(f"[{idx+1}/{len(cases)}] Running {case_id}...")
        run_mvp(args_ns, args_raw=args_raw, config_path=None, config_payload=run_args)

        out_dir = Path(str(run_args["out_dir"]))
        actual = _load_daily_report(out_dir)
        expected = _validate_expect(case.get("expect"))
        errors = _compare_expectations(expected, actual) if expected else []
        passed = not errors

        result = CaseResult(
            case_id=case_id,
            out_dir=str(out_dir),
            passed=passed,
            expected=expected,
            actual=actual,
            errors=errors,
        )
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"{status}: {case_id}")
        if errors:
            for err in errors:
                print(f"  - {err}")

        if errors and args.stop_on_fail:
            break

    summary = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [asdict(r) for r in results],
    }
    summary_path = Path(args.out_dir) / "golden_report.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote golden report: {summary_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
