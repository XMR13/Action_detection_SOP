"""Minimal website MVP (FastAPI) for reviewing SOP session artifacts.

This package is intentionally filesystem-first:
- Session artifacts are discovered from `data/sessions/.../session_*/`.
- SQLite stores only the review layer (decision, note, overrides).
"""

