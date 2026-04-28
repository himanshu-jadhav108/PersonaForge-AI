# PersonaForge AI — Model Registry & Licensing Record

This registry provides a single source of truth for every machine learning model used, supported, or evaluated in PersonaForge AI. It records source repositories, upstream licenses, commercial/redistribution terms, framework requirements, hardware footprints, and technical limitations in compliance with project safeguards.

---

## 1. Production Models (Active in Core Pipeline)

### InSwapper 128 (`inswapper_128.onnx`)
* **Model Name:** InSwapper 128 (InsightFace swap backbone)
* **Source / Origin:** InsightFace Research Team (`deepinsight/insightface`)
* **Framework:** ONNX Runtime (`onnxruntime`, `onnxruntime-gpu`)
* **Model Size:** ~529 MB
* **Input Resolution:** 128 × 128 pixels (aligned 5-point face crop)
* **Embeddings Input:** 512-dimensional ArcFace identity vector
* **Download Method:** Automated multi-mirror download in `models/model_manager.py` with sha256 verification and fallback mirrors (Hugging Face, GitHub mirror, Archive).
* **Hardware Requirements:**
  * CPU: ~500 MB RAM, runs via `CPUExecutionProvider`
  * GPU: ~1.5 GB dedicated VRAM via `CUDAExecutionProvider`
* **Licensing & Restrictions:**
  * **Code License:** MIT License (InsightFace repository code)
  * **Weight License:** InsightFace Non-Commercial Research License
  * **Commercial Use:** **RESTRICTED (Non-Commercial Research Only)**
  * **Redistribution:** Permitted for academic and open-source non-commercial research with attribution.
  * **Attribution:** InsightFace project (`https://github.com/deepinsight/insightface`).
* **Purpose:** Performs the core facial feature transfer in latent space between source identity embedding and target face crop.
* **Known Limitations:**
  * Fixed 128×128 pixel resolution output requires blending and optional restoration for high-resolution target videos.
  * Strong yaw angles (>60°) can suffer from latent distortion.

---

### InsightFace Buffalo_L Suite
* **Model Name:** Buffalo_L Detection, Alignment, and Recognition Suite
* **Components:**
  * `det_10g.onnx` — SCRFD face detector (10G FLOPs model)
  * `w600k_r50.onnx` — ArcFace ResNet50 512D recognition feature extractor
  * `2d106det.onnx` — 2D 106-point landmark localization model
  * `1k3d68.onnx` — 3D 68-point landmark model
  * `genderage.onnx` — Gender and age attribute estimation model
* **Source / Origin:** InsightFace (`deepinsight/insightface`)
* **Framework:** ONNX Runtime
* **Download Method:** Automated via InsightFace package (`FaceAnalysis(name='buffalo_l')`) to `~/.insightface/models/buffalo_l/`.
* **Hardware Requirements:**
  * CPU: ~1.2 GB RAM
  * GPU: ~1.0 GB VRAM
* **Licensing & Restrictions:**
  * **License:** InsightFace Non-Commercial Research License
  * **Commercial Use:** **RESTRICTED (Non-Commercial Research Only)**
  * **Redistribution:** Subject to InsightFace research terms.
* **Purpose:**
  * Multi-face detection and bounding box localization.
  * Accurate 5-point and 106-point facial landmark alignment.
  * Extraction of canonical 512D identity vectors for identity matching and consistency tracking.
* **Known Limitations:**
  * Heavily occluded faces (e.g., masks, hands covering chin/eyes) may drop detection confidence below threshold.

---

## 2. Restoration Models & Fallbacks (Evaluated for Phase 6)

### Primary Candidate: GFPGAN v1.4 (`GFPGANv1.4.pth`)
* **Model Name:** GFPGAN v1.4 (Generative Facial Prior GAN)
* **Source / Origin:** Tencent ARC Lab (`TencentARC/GFPGAN`)
* **Framework:** PyTorch (`torch`, `torchvision`) / ONNX (under verification)
* **Model Size:** ~332 MB
* **Hardware Requirements:**
  * CPU: ~1.5 GB RAM (high latency)
  * GPU: ~2.0 GB dedicated VRAM
* **Licensing & Restrictions:**
  * **Code License:** Apache License 2.0
  * **Model Weights License:** Non-commercial research / Apache 2.0 (Official repo licensed under Apache 2.0).
  * **Commercial Use:** Permitted under Apache 2.0 with appropriate attribution, subject to Tencent ARC terms.
  * **Redistribution:** Permitted under Apache 2.0 terms.
* **Purpose:** Blind facial restoration and fidelity enhancement on swapped crops.
* **Known Limitations:**
  * May synthesize micro-textures (teeth, eye reflections) that slightly alter identity if restoration weight is set too high ($w > 0.7$).

---

### Secondary Candidate: CodeFormer (`codeformer.pth`)
* **Model Name:** CodeFormer (Robust Face Restoration via Codebook Lookup Transformer)
* **Source / Origin:** SCUT-SPAL / Shangchen Zhou (`sczhou/CodeFormer`)
* **Framework:** PyTorch / TorchScript
* **Model Size:** ~375 MB
* **Hardware Requirements:**
  * CPU: ~2.0 GB RAM
  * GPU: ~2.5 GB dedicated VRAM
* **Licensing & Restrictions:**
  * **Code License:** S-Lab License 1.0 (Non-commercial research only)
  * **Model Weights License:** Strictly Non-Commercial Research
  * **Commercial Use:** **PROHIBITED without explicit license from NTU/S-Lab**
  * **Redistribution:** Non-commercial only with full attribution.
* **Purpose:** High-fidelity facial restoration with adjustable codebook weighting ($w$).
* **Known Limitations:**
  * S-Lab license prevents commercial redistribution.
  * High dependency footprint in PyTorch environments.

---

### Classical Enhancement (Deterministic Fallback)
* **Method:** Adaptive Bilateral Filtering + Unsharp Masking + CLAHE
* **Source / Origin:** OpenCV Core (`cv2`)
* **Framework:** Native OpenCV / NumPy (C++ / Python)
* **Hardware Requirements:** Negligible CPU footprint, zero extra VRAM.
* **Licensing:** Apache 2.0 (OpenCV license)
* **Transparency Safeguard:**
  * **STRICT RULE:** Classical enhancement must **never** be presented or labeled as AI face restoration.
  * When AI weights (GFPGAN/CodeFormer) are unavailable, the engine reports:
    ```
    AI Restoration: Unavailable
    Classical Enhancement: Available / Active
    ```
  * Every processing report logs `restoration_method: "none" | "classical" | "gfpgan" | "codeformer"`.

---

## 3. License Compliance Matrix

| Model / Component | Source Code License | Weights License | Commercial Use | Redistribution | Required Attribution |
|---|---|---|---|---|---|
| `inswapper_128.onnx` | MIT (Repo) | InsightFace Non-Commercial | ❌ No | Non-commercial | InsightFace citation |
| `buffalo_l` suite | MIT (Repo) | InsightFace Non-Commercial | ❌ No | Non-commercial | InsightFace citation |
| `GFPGAN v1.4` | Apache 2.0 | Apache 2.0 / Tencent | ⚠️ Verify terms | Permitted (Apache 2.0) | Tencent ARC citation |
| `CodeFormer` | S-Lab 1.0 | S-Lab Non-Commercial | ❌ No | Non-commercial only | SCUT / S-Lab citation |
| Classical OpenCV | Apache 2.0 | N/A (Algorithmic) | ✅ Yes | Permitted (Apache 2.0) | OpenCV citation |
