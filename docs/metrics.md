# PersonaForge AI — Metrics and Evaluation Standards

This document establishes the mathematical definitions, calibration standards, normalization formulas, and operational limitations for all quality, identity, and integrity metrics in PersonaForge AI.

In accordance with PersonaForge safeguards:
* No arbitrary scientific-looking percentages are generated.
* Metrics without validated algorithms (such as uncomputed head-pose yaw/pitch penalties) are strictly excluded until formally implemented.
* The composite score is named **PersonaForge Integrity Score** (or **Processing Score**), avoiding probabilistic terms like "Confidence Score".

---

## 1. Identity Consistency Metric

### ArcFace Cosine Similarity ($S_{cos}$)
* **Definition:** Measures cosine similarity between the 512-dimensional canonical identity embedding vector of the source face and the detected face in the swapped frame.
* **Formula:**
  $$S_{cos} = \frac{\mathbf{e}_{src} \cdot \mathbf{e}_{dst}}{\|\mathbf{e}_{src}\|_2 \|\mathbf{e}_{dst}\|_2}$$
* **Inputs:** Normalized feature vectors $\mathbf{e} \in \mathbb{R}^{512}$ produced by the `w600k_r50.onnx` ArcFace model.
* **Raw Metric Range:** $[-1.0, 1.0]$. In facial verification practice, random pairs typically measure between $-0.10$ and $+0.25$, while same-identity matches range between $+0.40$ and $+0.85$.
* **Calibration & Normalization:**
  $$\text{NormSim} = \text{clip}\left(\frac{S_{cos} - \tau_{baseline}}{\tau_{high} - \tau_{baseline}}, 0.0, 1.0\right) \times 100$$
  where $\tau_{baseline} = 0.20$ (random face boundary) and $\tau_{high} = 0.70$ (clean matching threshold).
* **Operational Thresholds:**
  * $\ge 0.50$ (Norm: $\ge 60\%$): Strong identity preservation.
  * $0.35 - 0.49$ (Norm: $30\% - 59\%$): Acceptable transfer; mild angle or lighting shift.
  * $< 0.35$ (Norm: $< 30\%$): Weak match or identity drift warning.
* **Known Limitations:**
  * Severe profile angles ($>45^\circ$) naturally lower raw ArcFace similarity even when identity is preserved.
  * Extreme expressions (wide smile, yawn) compress the feature vector distance.

---

## 2. Sharpness and Detail Metrics

### Laplacian Edge Variance ($\sigma_L^2$)
* **Definition:** Evaluates high-frequency spatial variation across the swapped face crop to identify motion blur, out-of-focus blurs, or interpolation smearing.
* **Formula:**
  $$\sigma_L^2 = \frac{1}{M \cdot N}\sum_{x=1}^M \sum_{y=1}^N \left( \nabla^2 I(x,y) - \mu_L \right)^2$$
  where $\nabla^2 I = \text{cv2.Laplacian}(I_{gray}, \text{cv2.CV\_64F})$ and $\mu_L$ is the mean response.
* **Inputs:** Grayscale 8-bit face crop $I_{gray}$.
* **Raw Metric Range:** $[0, \infty)$, practically $[5, 1200]$ for video face crops.
* **Calibration & Normalization:**
  Logarithmic scaling prevents outlier sharp frames from skewing evaluation:
  $$\text{NormSharp} = \text{clip}\left(\frac{\ln(1 + \sigma_L^2) - \ln(1 + \sigma_{min}^2)}{\ln(1 + \sigma_{target}^2) - \ln(1 + \sigma_{min}^2)}, 0.0, 1.0\right) \times 100$$
  where $\sigma_{min}^2 = 25.0$ (unusable blur) and $\sigma_{target}^2 = 300.0$ (standard high-definition sharpness).
* **Thresholds:**
  * $< 40$: Noticeable blur / degraded crop.
  * $40 - 150$: Standard acceptable sharpness.
  * $> 150$: Sharp, high-frequency edge definition.
* **Known Limitations:**
  * Sensor noise or film grain inflates high-frequency variance without contributing true structural sharpness.

---

## 3. Boundary and Coherence Metrics

### Boundary Gradient Smoothness ($G_{boundary}$)
* **Definition:** Assesses the transition across the blended boundary seam between the inserted face patch and the background target frame.
* **Formula:**
  $$G_{boundary} = \frac{1}{|\mathcal{B}|}\sum_{(x,y) \in \mathcal{B}} \|\nabla I(x, y)\|_2$$
  where $\mathcal{B}$ denotes the 5-pixel dilated boundary perimeter around the swap mask.
* **Inputs:** The composite frame and the alpha-blending mask perimeter.
* **Calibration:** Compared relative to the inner face gradient and adjacent background gradient to detect hard edge stitching artifacts.
* **Thresholds:** Sharp step changes exceeding $2.5\times$ local gradient denote unblended seams.
* **Known Limitations:** Hair or eyeglasses crossing the perimeter boundary naturally have elevated gradients and must not be falsely penalized.

---

## 4. Temporal Alignment and Landmark Stability

### Normalized Landmark Jitter ($\Delta_{landmark}$)
* **Definition:** Mean inter-frame Euclidean displacement of key facial landmarks, normalized by the inter-ocular distance (IOD).
* **Formula:**
  $$\Delta_{jitter} = \frac{1}{K}\sum_{k=1}^K \frac{\|\mathbf{p}_k^{(t)} - \mathbf{p}_k^{(t-1)}\|_2}{\text{IOD}^{(t)}}$$
  where $\text{IOD} = \|\mathbf{p}_{right\_eye} - \mathbf{p}_{left\_eye}\|_2$.
* **Inputs:** 5-point canonical facial landmarks between consecutive frames $t-1$ and $t$.
* **Calibration:**
  Normalizing by IOD makes displacement invariant to camera zoom, crop scale, and video resolution.
* **Thresholds:**
  * $< 0.05$ IOD: Smooth temporal movement.
  * $0.05 - 0.15$ IOD: Normal head movement or velocity.
  * $> 0.18$ IOD: High-frequency jitter or landmark tracking snap.
* **Known Limitations:**
  * Rapid camera pans or sudden cuts create natural displacement spikes; require scene-change detection masking.

---

## 5. Composite Metric: PersonaForge Integrity Score

* **Terminology Note:** Renamed from "Confidence Score". The metric is an explainable weighted heuristic composite, not a statistical Bayesian confidence probability.
* **Components and Weighting:**
  $$\text{Integrity Score} = 0.40 \cdot \text{NormSim} + 0.35 \cdot \text{NormSharp} + 0.25 \cdot (100 - \text{NormJitter})$$
* **Excluded Heuristics (Safeguard 7):**
  * Yaw / Pitch / Roll pose penalties are **strictly excluded** until an explicit, calibrated 3D `solvePnP` head-pose estimator is implemented and validated against ground-truth angles.
* **Output:**
  * Returns an integer score $[0, 100]$.
  * Always provides individual sub-score breakdowns in the inspection report so developers and users can see exactly why a score was assigned.
