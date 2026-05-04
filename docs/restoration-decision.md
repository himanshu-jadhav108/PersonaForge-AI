# Technical Decision Record: Face Restoration Strategy (TDR-002)

## Status
**ACCEPTED** (Implementation in Phase 6)

## Context & Motivation
Face swapping models (such as `inswapper_128.onnx`) operate at fixed low-resolution latent patches ($128 \times 128$), which often produces slight texture softening or eye artifacting when upscaled onto high-definition target video frames ($720\text{p}$, $1080\text{p}$). To achieve portfolio-grade visual fidelity, a face restoration enhancement step is desirable.

However, face restoration introduces significant technical challenges:
1. **License Restrictions:** Some popular models (e.g., CodeFormer) carry restrictive non-commercial research-only licenses.
2. **Framework & Dependency Overhead:** Official PyTorch/torchvision implementations add $700\text{MB} - 1.5\text{GB}$ of binary dependencies and heavy CUDA runtime overhead.
3. **Identity Drift:** Generative face restoration often alters facial features (eye shape, lip curvature, nose bridge) toward a generic statistical prior, degrading identity preservation.
4. **Transparency Mandate (Safeguard 2):** Classical image processing filters (bilateral filtering, unsharp masking) must never be deceptively reported to users as AI restoration.

---

## Architectural Comparison Matrix

| Evaluation Dimension | GFPGAN v1.4 (TencentARC) | CodeFormer (S-Lab / NTU) | Classical Filters (OpenCV) |
|---|---|---|---|
| **Model Source** | Official TencentARC GitHub / HuggingFace | S-Lab NTU GitHub / HuggingFace | Native OpenCV (Bilateral + Unsharp) |
| **License** | **Apache 2.0 (Permissive Commercial)** | **Non-Commercial Research Only (S-Lab 1.0)** | **Apache 2.0 (OpenCV Native)** |
| **Primary Framework** | ONNX Runtime / PyTorch | PyTorch / ONNX (custom ops) | Pure C++ / Python (OpenCV) |
| **Dependency Overhead** | Low (if ONNX) / Moderate (if PyTorch) | High (`torch`, `torchvision`, `basicsr`) | **Zero extra dependencies** |
| **VRAM / RAM Impact** | ~1.2 GB VRAM (FP16) / ~800 MB RAM | ~1.8 GB VRAM (FP16) / ~1.2 GB RAM | < 50 MB RAM |
| **Processing Speed** | ~40-60 ms per $512\times512$ crop | ~75-120 ms per $512\times512$ crop | **~2-5 ms per crop** |
| **Identity Preservation** | **High** (Facial Prior with identity loss) | High (Adjustable $w$ trade-off) | Neutral (Preserves original pixels) |
| **Artifact Removal** | Excellent for skin & eye textures | Superior for severe occlusion | Mild softening only |
| **Integration Complexity** | Moderate (standard network topology) | High (codebook lookup layer) | Low |

---

## Architectural Decisions

### 1. Primary AI Restorer: GFPGAN v1.4 (Apache 2.0)
* **Rationale:** GFPGAN is published under the permissive **Apache 2.0 license**, making it fully compliant for open-source and commercial use without restrictive research-only covenants.
* **Execution Boundary:** Restoration is executed strictly on the $512 \times 512$ aligned face crop, never on full video frames. This bounds compute and memory footprint to a constant $O(1)$ per face.

### 2. Optional Research Adapter: CodeFormer
* **Rationale:** Provided as a modular plug-in adapter for non-commercial research comparative evaluation, clearly guarded by a license warning.

### 3. Restoration Identity Guard (`RestorationIdentityGuard`)
* **Problem:** Generative restorations risk smoothing out distinctive source identity markers (moles, eye asymmetry, distinctive nasal angles).
* **Mitigation:**
  - The guard extracts ArcFace embeddings immediately before and after restoration.
  - If cosine similarity between the restored face and source face drops by $\Delta \ge 0.08$ compared to the pre-restored face, an `IDENTITY_DRIFT_WARNING` is logged.
  - The system dynamically throttles the restoration blending factor (e.g., from $1.0$ down to $0.5$) to protect source identity fidelity.

### 4. Transparent Fallback (Safeguard 2 Compliance)
When AI restoration weights are unavailable on the local machine:
* The system logs and reports explicitly:
  ```json
  {
    "ai_restoration": "Unavailable",
    "classic_enhancement": "Enabled (Bilateral / Unsharp Mask)",
    "status": "Running classical spatial enhancement without generative restoration"
  }
  ```
* Classical filtering is never labeled as "AI Restoration" in API responses, log files, or UI indicators.
