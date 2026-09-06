"""
backend/app/security/sanitization.py — Input validation, path-traversal prevention, and media signature verification.
"""

from __future__ import annotations

import re
from pathlib import Path

# Match 32-character hex string (standard hex UUID without hyphens) or standard 36-char UUID
SESSION_ID_PATTERN = re.compile(
    r"^(?:[a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$",
    re.IGNORECASE,
)

# Safe filename character whitelist
SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_\-\.]")


def validate_session_id(session_id: str | None) -> bool:
    """Validate session_id to prevent directory traversal, injection, or unexpected path elements.

    Args:
        session_id: Candidate session identifier string.

    Returns:
        True if the session_id is a valid UUID, False otherwise.
    """
    if not session_id or not isinstance(session_id, str):
        return False

    # Check for explicit traversal tokens or null bytes
    if ".." in session_id or "/" in session_id or "\\" in session_id or "\x00" in session_id:
        return False

    return bool(SESSION_ID_PATTERN.match(session_id.strip()))


def sanitize_filename(filename: str, default: str = "unnamed_file") -> str:
    """Sanitize a user-provided filename by removing path traversal characters and directory parts.

    Args:
        filename: Raw user filename.
        default: Fallback name if sanitized string becomes empty.

    Returns:
        A safe filename with only permitted alphanumeric, underscore, dot, and hyphen characters.
    """
    if not filename or not isinstance(filename, str):
        return default

    # Extract basename only to discard any preceding directory traversal tokens
    base = Path(filename).name.strip()
    if not base or base in (".", ".."):
        return default

    # Remove non-safe characters
    cleaned = SAFE_FILENAME_PATTERN.sub("_", base)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")

    return cleaned if cleaned else default


def validate_media_magic_bytes(file_path: Path | str, media_type: str) -> bool:
    """Verify file magic bytes against expected binary file signatures.

    Args:
        file_path: Path to the target media file on disk.
        media_type: 'image' or 'video'.

    Returns:
        True if header matches valid magic bytes for the declared media type.
    """
    p = Path(file_path)
    if not p.is_file() or p.stat().st_size < 12:
        return False

    try:
        with p.open("rb") as f:
            header = f.read(32)

        if media_type == "image":
            # JPEG: FF D8 FF
            if header.startswith(b"\xff\xd8\xff"):
                return True
            # PNG: 89 50 4E 47 0D 0A 1A 0A
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                return True
            # WEBP: RIFF .... WEBP
            return header.startswith(b"RIFF") and header[8:12] == b"WEBP"

        if media_type == "video":
            # MP4 / MOV: check for ftyp box or moov/mdat
            if b"ftyp" in header[:16] or b"moov" in header[:16] or b"mdat" in header[:16]:
                return True
            # Matroska / MKV / WebM: 1A 45 DF A3
            if header.startswith(b"\x1a\x45\xdf\xa3"):
                return True
            # AVI: RIFF .... AVI
            return header.startswith(b"RIFF") and header[8:12] == b"AVI "

    except (OSError, PermissionError):
        return False

    return False


def validate_file_security(
    file_path: Path | str,
    media_type: str,
    max_bytes: int | None = None,
) -> tuple[bool, str]:
    """Perform combined size and magic byte verification on uploaded media.

    Args:
        file_path: Disk path to the candidate file.
        media_type: Expected media type ('image' or 'video').
        max_bytes: Optional size limit in bytes.

    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    p = Path(file_path)
    if not p.exists():
        return False, f"File does not exist: {p.name}"

    size = p.stat().st_size
    if size == 0:
        return False, "File is empty (0 bytes)"

    if max_bytes is not None and size > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return False, f"File size ({size / (1024 * 1024):.1f}MB) exceeds limit of {limit_mb:.0f}MB"

    if not validate_media_magic_bytes(p, media_type):
        return False, f"Invalid {media_type} signature or unsupported binary header"

    return True, "Valid"
