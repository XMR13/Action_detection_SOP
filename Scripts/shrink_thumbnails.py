"""One-time backfill: re-encode existing full-resolution session thumbnails.

Dry-run by default. With --apply each ``sessions/<date>/<session>/thumbnail.jpg``
is downscaled/recompressed in place (atomic rename) only when the result is
smaller. The original mtime, permissions and owner are preserved so retention
age checks and the web service's file access are unaffected.

Example (on the Jetson):
    python3 -m Scripts.shrink_thumbnails --data-dir /var/lib/action-sop/roll_sop_v1
    sudo python3 -m Scripts.shrink_thumbnails --data-dir /var/lib/action-sop/roll_sop_v1 --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from Action_Detection_SOP.thumbnails import ThumbnailSpec, atomic_write_bytes, encode_thumbnail


@dataclass(frozen=True)
class ShrinkSummary:
    dry_run: bool
    scanned: int
    shrunk: int
    skipped_already_small: int
    failed: int
    bytes_before: int
    bytes_after: int

    @property
    def bytes_saved(self) -> int:
        return self.bytes_before - self.bytes_after


def _preserve_metadata(path: Path, st: os.stat_result) -> None:
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.chmod(path, st.st_mode & 0o7777)
    try:
        os.chown(path, st.st_uid, st.st_gid)
    except PermissionError:
        pass  # non-root run: file is already owned by the current user


def shrink_thumbnails(
    *,
    sessions_dir: Path,
    spec: ThumbnailSpec,
    dry_run: bool,
    min_size_bytes: int = 150_000,
    progress_every: int = 0,
) -> ShrinkSummary:
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"sessions directory not found: {sessions_dir}")
    scanned = shrunk = skipped = failed = 0
    before = after = 0
    paths = sorted(sessions_dir.glob("*/*/thumbnail.jpg"))
    total = len(paths)
    for path in paths:
        scanned += 1
        if progress_every > 0 and scanned % progress_every == 0:
            print(
                f"[shrink_thumbnails] {scanned}/{total} scanned, {shrunk} to shrink, "
                f"{(before - after) / 1e9:.2f} GB saved so far",
                file=sys.stderr,
                flush=True,
            )
        try:
            st = path.stat()
            before += st.st_size
            if st.st_size < min_size_bytes:
                skipped += 1
                after += st.st_size
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("unreadable JPEG")
            if image.shape[1] <= spec.max_width:
                # Already thumbnail-sized; re-encoding would only add generational JPEG loss.
                skipped += 1
                after += st.st_size
                continue
            data = encode_thumbnail(image, spec)
            if len(data) >= st.st_size:
                skipped += 1
                after += st.st_size
                continue
            if not dry_run:
                atomic_write_bytes(path, data)
                _preserve_metadata(path, st)
            shrunk += 1
            after += len(data)
        except (OSError, ValueError, RuntimeError) as e:
            failed += 1
            after += st.st_size if "st" in locals() else 0
            print(json.dumps({"error": str(e), "path": str(path)}))
    return ShrinkSummary(
        dry_run=dry_run,
        scanned=scanned,
        shrunk=shrunk,
        skipped_already_small=skipped,
        failed=failed,
        bytes_before=before,
        bytes_after=after,
    )


def main(argv: Optional[List[str]] = None) -> int:
    defaults = ThumbnailSpec()
    parser = argparse.ArgumentParser(description="Downscale existing session thumbnails in place.")
    parser.add_argument("--data-dir", type=Path, required=True, help="SOP data dir, e.g. /var/lib/action-sop/roll_sop_v1")
    parser.add_argument("--max-width", type=int, default=defaults.max_width)
    parser.add_argument("--jpeg-quality", type=int, default=defaults.jpeg_quality)
    parser.add_argument(
        "--min-size-kb", type=int, default=150, help="Skip thumbnails already smaller than this (KB)."
    )
    parser.add_argument(
        "--progress-every", type=int, default=500, help="Print progress to stderr every N files (0 = off)."
    )
    parser.add_argument("--apply", action="store_true", help="Rewrite files. Default is dry-run.")
    args = parser.parse_args(argv)

    spec = ThumbnailSpec(max_width=args.max_width, jpeg_quality=args.jpeg_quality)
    summary = shrink_thumbnails(
        sessions_dir=args.data_dir / "sessions",
        spec=spec,
        dry_run=not args.apply,
        min_size_bytes=args.min_size_kb * 1000,
        progress_every=args.progress_every,
    )
    payload = {
        "data_dir": str(args.data_dir),
        "spec": asdict(spec),
        "summary": asdict(summary),
        "gb_before": round(summary.bytes_before / 1e9, 2),
        "gb_after": round(summary.bytes_after / 1e9, 2),
        "gb_saved": round(summary.bytes_saved / 1e9, 2),
    }
    print(json.dumps(payload, indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
