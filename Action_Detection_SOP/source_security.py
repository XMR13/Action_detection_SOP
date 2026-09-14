from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit


def redact_source_credentials(value: Optional[str]) -> Optional[str]:
    """Return a source label that cannot expose URL userinfo or query secrets."""
    if value is None:
        return None

    raw = str(value).strip()
    if "://" not in raw:
        return raw

    scheme_hint = raw.split("://", 1)[0].lower() or "source"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return f"{scheme_hint}://redacted-source"

    if not parsed.scheme or not parsed.netloc:
        return f"{scheme_hint}://redacted-source"

    # Userinfo may contain ':' or an unescaped '@', so keep only the final
    # netloc segment. Query strings and fragments can also carry credentials.
    safe_netloc = parsed.netloc.rsplit("@", 1)[-1]
    if not safe_netloc:
        return f"{parsed.scheme.lower()}://redacted-source"

    return urlunsplit((parsed.scheme.lower(), safe_netloc, parsed.path, "", ""))

def redact_source_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Copy a persisted/API payload while redacting known sourc fields.
    """

    out: Dict[str, Any] = dict(payload)
    # argparse/config snapshots store these fields at the top leve, while
    # persisted run metadata can also place them under ''sourcec'' or ''args''.

    for key in("rtsp", "video"):
        if isinstance(out.get(key), str):
            out[key] = redact_source_credentials(out[key])

    #getting the output source

    source = out.get("source")
    if isinstance(source, str):
        out["source"] = redact_source_credentials(source)

    elif isinstance(source, Mapping):
        source_out = dict(source)

        for key in ("rtsp", "video"):
            if isinstance(source_out.get(key), str):
                source_out[key] = redact_source_credentials(source_out[key])
        out["source"] = source_out

    if isinstance(out.get("camera_id"), str):
        out["camera_id"] = redact_source_credentials(out["camera_id"])

    args = out.get("args")
    if isinstance(args, Mapping):
        args_out = dict(args)
        for key in ("rtsp", "video"):
            if isinstance(args_out.get(key), str):
                args_out[key] = redact_source_credentials(args_out[key])

        out["args"] = args_out

    return out
