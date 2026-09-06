# Security Policy — PersonaForge AI

PersonaForge AI is committed to maintaining the security, integrity, and privacy of users running facial transformation software on local and server environments.

---

## 1. Supported Versions

Security patches and bug fixes are actively provided for the following versions:

| Version | Supported | Notes |
|---|---|---|
| **2.0.x** | :white_check_mark: | Active production line with complete security architecture. |
| < 2.0.0 | :x: | Legacy versions; users are advised to upgrade to 2.0.0+. |

---

## 2. Reporting a Vulnerability

If you discover a security vulnerability or privacy flaw within PersonaForge AI:

1. **Do NOT open a public GitHub issue.**
2. Please submit an advisory or report details via GitHub Security Advisories or email the maintainers directly.
3. Include:
   * A detailed description of the vulnerability.
   * Steps to reproduce or proof-of-concept (PoC) code.
   * Assessment of potential impact (e.g. arbitrary file read, directory traversal, resource exhaustion).
4. Maintainers will acknowledge your report within 48 hours and coordinate a coordinated disclosure timeline.

---

## 3. Core Security Controls

PersonaForge incorporates defensive mechanisms against common web and machine learning application vulnerabilities:

### A. Path Traversal & Injection Defense
* **UUID Whitelisting**: Session identifiers are validated strictly against regex patterns (`^[a-f0-9]{32}$` or 36-char hyphenated UUIDs). Tokens containing `..`, `/`, `\`, or null bytes `%00` are rejected with HTTP 400.
* **Filename Sanitization**: Uploaded filenames are stripped of path components and normalized to safe alphanumeric, underscore, dot, and hyphen characters.

### B. Upload Signature Inspection
* Media files are validated via binary magic bytes:
  * **Images**: JPEG (`FF D8 FF`), PNG (`89 50 4E 47`), WEBP (`RIFF....WEBP`).
  * **Videos**: MP4/MOV (`ftyp`, `moov`, `mdat`), Matroska/WebM (`1A 45 DF A3`), AVI (`RIFF....AVI`).
* Spoofed files (e.g., shell scripts or executables renamed to `.jpg`) are rejected immediately before disk staging.

### C. Resource Exhaustion & Streaming Protection
* Uploads are streamed in 64KB chunks to prevent RAM exhaustion, with hard upper bounds (50MB for images, 500MB for videos).
* Processing tasks are serialized via async semaphores to prevent CPU/GPU thrashing.

### D. Automated Data Retention & Privacy Purging
* Configurable retention policy governed by `RETENTION_HOURS` (default 24 hours).
* `RetentionManager` periodically purges aged uploads, temporary frames, and output assets from disk.
* On-demand cleanup is accessible to administrators via `POST /system/cleanup`.
