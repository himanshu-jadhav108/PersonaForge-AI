# PersonaForge AI — Identity Metrics & Explainable Confidence Scoring

PersonaForge AI employs a mathematically grounded, multi-component validation engine to evaluate identity preservation, boundary coherence, facial sharpness, and temporal landmark stability.

---

## 1. Mathematical Foundations

### A. ArcFace 512-Dimensional Cosine Similarity
For two facial embedding vectors $u, v \in \mathbb{R}^{512}$ extracted by the ArcFace ResNet50 model (`w600k_r50.onnx`), cosine similarity measures directional alignment:

$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \, \|v\|_2} = \frac{\sum_{i=1}^{512} u_i v_i}{\sqrt{\sum_{i=1}^{512} u_i^2} \sqrt{\sum_{i=1}^{512} v_i^2}}$$

Because ArcFace embeddings are normalized during inference ($\|u\|_2 = 1$), this reduces to the direct dot product:

$$\text{Cosine Similarity}(u, v) = u \cdot v \in [-1.0, 1.0]$$

In practical facial verification, scores $\ge 0.60$ indicate the same individual; scores $\ge 0.80$ represent high-fidelity preservation.

### B. Rolling Similarity & Sudden Drop Detection
To prevent temporary occlusions or rapid head rotations from skewing global scores, PersonaForge calculates a rolling moving average with window size $W = 10$:

$$\overline{S}_t = \frac{1}{\min(t, W)} \sum_{k=0}^{\min(t, W)-1} S_{t-k}$$

A **Sudden Drop Alert** is flagged whenever:

$$\Delta_{\text{drop}} = \overline{S}_{t-1} - S_t \ge 0.15$$

---

## 2. Identity Drift Classification Zones

PersonaForge categorizes identity preservation into three empirical operational zones:

| Drift Zone | Similarity Range | Meaning & Visual Indicators |
|---|---|---|
| **STABLE** | $S \ge 0.80$ | High identity consistency. Source identity is strongly preserved. Green badge. |
| **WARNING** | $0.65 \le S < 0.80$ | Moderate identity fidelity. Extreme pose angles, lighting shifts, or heavy expressions. Yellow badge. |
| **CRITICAL** | $S < 0.65$ | Severe identity degradation or target face occlusion. Requires re-assessment. Red badge. |

---

## 3. Physical & Visual Quality Metrics

### A. Normalized Sharpness (Sobel Gradient Density)
Evaluates gradient density on normalized facial crops:
$$\text{Sharpness} = \frac{1}{M \cdot N} \sum_{x, y} \left( |G_x(x,y)| + |G_y(x,y)| \right)$$
Calibrated between 0 and 100 via sigmoidal mapping.

### B. Edge Focus (Laplacian Variance)
Measures the variance of the 2D Laplacian operator $\nabla^2 I$ over the facial crop:
$$\text{Var}(\nabla^2 I) = \frac{1}{N} \sum (L(x,y) - \mu_L)^2$$
Higher variance indicates sharp edges and fine skin texture, while values $< 50.0$ indicate motion blur.

### C. Landmark Jitter Normalized by Inter-Ocular Distance (IOD)
Quantifies temporal flickering across consecutive frames:
$$\text{IOD} = \| P_{\text{left\_eye}} - P_{\text{right\_eye}} \|_2$$
$$\text{Jitter}_{\text{normalized}} = \frac{\frac{1}{68} \sum_{i=1}^{68} \| P_{i, t} - P_{i, t-1} \|_2}{\text{IOD}}$$
Values $< 0.04$ indicate smooth, jitter-free temporal tracking.

### D. Boundary Gradient Ratio
Evaluates the transition between the swapped face bounding polygon and background pixels:
$$R_{\text{boundary}} = \frac{\nabla_{\text{outer}}}{\nabla_{\text{inner}} + \epsilon}$$
Values near 1.0 indicate seamless natural blending without harsh cut lines.

---

## 4. PersonaForge Composite Integrity Score

The **PersonaForge Integrity Score** ($C_{\text{total}} \in [0, 100]$) combines the 5 core telemetry signals into an explainable weighted metric:

$$C_{\text{total}} = 100 \times \left( 0.40 \cdot C_{\text{identity}} + 0.20 \cdot C_{\text{sharpness}} + 0.15 \cdot C_{\text{stability}} + 0.15 \cdot C_{\text{boundary}} + 0.10 \cdot C_{\text{detection}} \right)$$

### Executive Scoring Tiers

| Tier | Score Range | Executive Recommendation |
|---|---|---|
| **EXCELLENT** | $85 - 100$ | Production-grade fidelity. Suitable for broadcast and high-definition video portfolios. |
| **GOOD** | $70 - 84$ | Solid visual fidelity with minor natural expression artifacts. Recommended for general usage. |
| **FAIR** | $55 - 69$ | Noticeable identity or boundary softening. Consider running with High Quality mode or AI Restoration. |
| **DEGRADED** | $< 55$ | Significant identity loss or tracking failures. Check lighting, source face resolution, or angle. |
