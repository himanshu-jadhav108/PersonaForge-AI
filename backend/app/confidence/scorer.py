import math

from backend.app.confidence.models import (
    ComponentScore,
    IntegrityBreakdown,
    IntegrityTier,
    PersonaForgeIntegrityReport,
)


class PersonaForgeIntegrityScorer:
    """
    Computes the explainable PersonaForge Integrity Score.
    Follows normalization and calibration standards defined in docs/metrics.md.
    Strictly excludes unverified pose penalties per Safeguard 7.
    """

    # Baseline & High match constants from docs/metrics.md
    SIM_BASELINE: float = 0.20
    SIM_HIGH: float = 0.70

    # Blur constants
    SIGMA_MIN_SQ: float = 25.0
    SIGMA_TARGET_SQ: float = 300.0

    # Jitter constant
    JITTER_MAX_IOD: float = 0.20

    # Boundary constant
    SEAM_MAX_RATIO: float = 2.5

    @classmethod
    def compute_norm_sim(cls, cosine_sim: float) -> float:
        """
        Normalizes raw cosine similarity [-1.0, 1.0] to [0.0, 100.0] scale.
        NormSim = clip((S_cos - 0.20) / (0.70 - 0.20), 0.0, 1.0) * 100.0
        """
        sim = float(cosine_sim)
        norm = (sim - cls.SIM_BASELINE) / (cls.SIM_HIGH - cls.SIM_BASELINE)
        return round(float(max(0.0, min(1.0, norm)) * 100.0), 2)

    @classmethod
    def compute_norm_sharp(cls, laplacian_var: float) -> float:
        """
        Logarithmically scales Laplacian variance to [0.0, 100.0] scale.
        """
        var = float(laplacian_var)
        if var <= 0.0:
            return 0.0
        log_var = math.log(1.0 + var)
        log_min = math.log(1.0 + cls.SIGMA_MIN_SQ)
        log_target = math.log(1.0 + cls.SIGMA_TARGET_SQ)
        if log_target <= log_min:
            return 0.0
        norm = (log_var - log_min) / (log_target - log_min)
        return round(float(max(0.0, min(1.0, norm)) * 100.0), 2)

    @classmethod
    def compute_norm_stability(cls, jitter_iod: float) -> float:
        """
        Normalizes landmark jitter in IOD units to [0.0, 100.0] scale.
        NormStability = clip(1.0 - (jitter / 0.20), 0.0, 1.0) * 100.0
        """
        j = float(jitter_iod)
        norm = 1.0 - (j / cls.JITTER_MAX_IOD)
        return round(float(max(0.0, min(1.0, norm)) * 100.0), 2)

    @classmethod
    def compute_norm_boundary(cls, boundary_ratio: float) -> float:
        """
        Normalizes boundary gradient transition ratio to [0.0, 100.0] scale.
        ratio <= 1.0 -> 100.0; ratio >= 2.5 -> 0.0
        """
        r = float(boundary_ratio)
        if r <= 1.0:
            return 100.0
        if r >= cls.SEAM_MAX_RATIO:
            return 0.0
        fraction = (r - 1.0) / (cls.SEAM_MAX_RATIO - 1.0)
        return round(float((1.0 - fraction) * 100.0), 2)

    @classmethod
    def compute_norm_detection(cls, det_score: float) -> float:
        """
        Normalizes raw detector confidence [0.0, 1.0] to [0.0, 100.0].
        """
        score = float(det_score)
        if score <= 1.0:
            score *= 100.0
        return round(float(max(0.0, min(100.0, score))), 2)

    @classmethod
    def evaluate(
        cls,
        job_id: str,
        cosine_similarity: float,
        laplacian_variance: float,
        jitter_iod: float = 0.02,
        boundary_ratio: float | None = None,
        det_score: float | None = None,
    ) -> PersonaForgeIntegrityReport:
        """
        Evaluates the full PersonaForge Integrity Report with explainable component scores.
        """
        norm_sim = cls.compute_norm_sim(cosine_similarity)
        norm_sharp = cls.compute_norm_sharp(laplacian_variance)
        norm_stab = cls.compute_norm_stability(jitter_iod)

        has_boundary = boundary_ratio is not None
        has_detection = det_score is not None

        # Determine weights based on available components
        if has_boundary and has_detection:
            w_sim, w_sharp, w_stab, w_bound, w_det = 0.35, 0.25, 0.20, 0.10, 0.10
        elif has_boundary:
            w_sim, w_sharp, w_stab, w_bound, w_det = 0.38, 0.30, 0.22, 0.10, 0.0
        elif has_detection:
            w_sim, w_sharp, w_stab, w_bound, w_det = 0.38, 0.30, 0.22, 0.0, 0.10
        else:
            # Baseline formula from docs/metrics.md Section 5: 0.40, 0.35, 0.25
            w_sim, w_sharp, w_stab, w_bound, w_det = 0.40, 0.35, 0.25, 0.0, 0.0

        # Status & Explanations: Identity
        if norm_sim >= 75.0:
            id_status = "optimal"
            id_exp = f"Strong identity retention ({cosine_similarity:.3f} cosine similarity against source face)."
        elif norm_sim >= 45.0:
            id_status = "acceptable"
            id_exp = (
                f"Moderate identity match ({cosine_similarity:.3f} similarity); minor yaw angle or lighting divergence."
            )
        else:
            id_status = "warning"
            id_exp = f"Low identity similarity ({cosine_similarity:.3f}); potential facial drift or landmark tracking failure."

        # Status & Explanations: Sharpness
        if norm_sharp >= 75.0:
            sharp_status = "optimal"
            sharp_exp = f"Crisp high-frequency edge definition (Laplacian variance {laplacian_variance:.1f})."
        elif norm_sharp >= 45.0:
            sharp_status = "acceptable"
            sharp_exp = f"Acceptable sharpness (Laplacian variance {laplacian_variance:.1f}); standard focus."
        else:
            sharp_status = "warning"
            sharp_exp = f"Noticeable blur detected (Laplacian variance {laplacian_variance:.1f} < 40)."

        # Status & Explanations: Temporal Stability
        if norm_stab >= 80.0:
            stab_status = "optimal"
            stab_exp = f"Smooth temporal alignment ({jitter_iod:.3f} IOD displacement); zero high-frequency jitter."
        elif norm_stab >= 50.0:
            stab_status = "acceptable"
            stab_exp = f"Moderate head movement ({jitter_iod:.3f} IOD displacement)."
        else:
            stab_status = "warning"
            stab_exp = f"Elevated landmark jitter ({jitter_iod:.3f} IOD > 0.18 threshold); tracking snap detected."

        id_comp = ComponentScore(
            name="Identity Preservation",
            raw_value=round(float(cosine_similarity), 4),
            raw_unit="cosine",
            normalized_score=norm_sim,
            weight=w_sim,
            weighted_contribution=round(norm_sim * w_sim, 2),
            status=id_status,
            explanation=id_exp,
        )

        sharp_comp = ComponentScore(
            name="Sharpness & Focus",
            raw_value=round(float(laplacian_variance), 2),
            raw_unit="variance",
            normalized_score=norm_sharp,
            weight=w_sharp,
            weighted_contribution=round(norm_sharp * w_sharp, 2),
            status=sharp_status,
            explanation=sharp_exp,
        )

        stab_comp = ComponentScore(
            name="Temporal Stability",
            raw_value=round(float(jitter_iod), 4),
            raw_unit="IOD",
            normalized_score=norm_stab,
            weight=w_stab,
            weighted_contribution=round(norm_stab * w_stab, 2),
            status=stab_status,
            explanation=stab_exp,
        )

        boundary_comp = None
        if has_boundary:
            norm_b = cls.compute_norm_boundary(boundary_ratio)
            b_status = "optimal" if norm_b >= 75.0 else ("acceptable" if norm_b >= 45.0 else "warning")
            b_exp = (
                f"Seamless edge transition (gradient ratio {boundary_ratio:.2f})."
                if norm_b >= 75.0
                else f"Boundary step change ratio {boundary_ratio:.2f}; possible seam artifact."
            )
            boundary_comp = ComponentScore(
                name="Boundary Coherence",
                raw_value=round(float(boundary_ratio), 3),
                raw_unit="ratio",
                normalized_score=norm_b,
                weight=w_bound,
                weighted_contribution=round(norm_b * w_bound, 2),
                status=b_status,
                explanation=b_exp,
            )

        det_comp = None
        if has_detection:
            norm_d = cls.compute_norm_detection(det_score)
            d_status = "optimal" if norm_d >= 85.0 else ("acceptable" if norm_d >= 65.0 else "warning")
            d_exp = f"Face detection confidence of {norm_d:.1f}%."
            det_comp = ComponentScore(
                name="Detection Reliability",
                raw_value=round(float(det_score), 4),
                raw_unit="confidence",
                normalized_score=norm_d,
                weight=w_det,
                weighted_contribution=round(norm_d * w_det, 2),
                status=d_status,
                explanation=d_exp,
            )

        # Composite score
        total_score = id_comp.weighted_contribution + sharp_comp.weighted_contribution + stab_comp.weighted_contribution
        if boundary_comp:
            total_score += boundary_comp.weighted_contribution
        if det_comp:
            total_score += det_comp.weighted_contribution

        final_score = round(max(0.0, min(100.0, total_score)), 2)

        # Tier assignment
        if final_score >= 85.0:
            tier = IntegrityTier.EXCELLENT
        elif final_score >= 70.0:
            tier = IntegrityTier.GOOD
        elif final_score >= 50.0:
            tier = IntegrityTier.FAIR
        else:
            tier = IntegrityTier.DEGRADED

        # Badges
        badges: list[str] = []
        if norm_sim >= 80.0:
            badges.append("IDENTITY_FIDELITY_HIGH")
        if norm_sharp >= 75.0:
            badges.append("STUDIO_SHARPNESS")
        if norm_stab >= 85.0:
            badges.append("TEMPORALLY_STABLE")
        if boundary_comp and boundary_comp.normalized_score >= 80.0:
            badges.append("SEAMLESS_BLEND")
        if det_comp and det_comp.normalized_score >= 90.0:
            badges.append("HIGH_DETECTION_CONFIDENCE")

        # Warning badges
        if norm_sim < 45.0:
            badges.append("DRIFT_RISK")
        if norm_sharp < 40.0:
            badges.append("MOTION_BLUR")
        if norm_stab < 60.0:
            badges.append("LANDMARK_JITTER")
        if boundary_comp and boundary_comp.normalized_score < 45.0:
            badges.append("SEAM_ARTIFACT")
        if det_comp and det_comp.normalized_score < 65.0:
            badges.append("OCCLUSION_WARNING")

        explanations = [id_exp, sharp_exp, stab_exp]
        if boundary_comp:
            explanations.append(boundary_comp.explanation)
        if det_comp:
            explanations.append(det_comp.explanation)

        recommendations: list[str] = []
        if norm_sim < 60.0:
            recommendations.append(
                "Source identity preservation is suboptimal. Consider a higher-resolution frontal source portrait."
            )
        if norm_sharp < 50.0:
            recommendations.append(
                "Face output exhibits blurriness. Processing in 'high' mode or applying restoration will sharpen details."
            )
        if norm_stab < 70.0:
            recommendations.append(
                "Video shows landmark jitter. Enable correlation filter tracking or stabilize the source footage."
            )
        if boundary_comp and boundary_comp.normalized_score < 60.0:
            recommendations.append(
                "Seam boundary transition is noticeable. Use 'FeatheredBlend' with larger blur radius."
            )

        if not recommendations:
            recommendations.append("Overall integrity is excellent! Pipeline generated studio-grade output.")

        breakdown = IntegrityBreakdown(
            identity=id_comp,
            sharpness=sharp_comp,
            temporal_stability=stab_comp,
            boundary_coherence=boundary_comp,
            detection_confidence=det_comp,
        )

        return PersonaForgeIntegrityReport(
            job_id=job_id,
            integrity_score=final_score,
            tier=tier,
            breakdown=breakdown,
            badges=badges,
            explanations=explanations,
            recommendations=recommendations,
        )
