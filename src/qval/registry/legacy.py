"""Compatibility tables for pre-convention on-disk artifacts.

These map *historical* prediction file names onto the current canonical method
names. They are deliberately **not** derived from :mod:`qval.registry.catalog`:
the catalog describes the current experiment matrix, whereas these describe old
files that predate the naming convention and merely need to keep resolving in
the paper-figure loader. Delete an entry only once the corresponding files are
renamed or removed on disk.

The paper-figure registry (``scripts/paper_figures/registry.py``) derives its
canonical tables from the catalog and consults these tables for the legacy
fallbacks; see :func:`scripts.paper_figures.registry.resolve_method`.
"""

from __future__ import annotations

import re


# Pre-convention ``method_name`` -> canonical ``method_name``.
# ``resolve_method`` recurses on the canonical name, which the paper-figure
# registry derives from the catalog. Delete this table (and the legacy half of
# ``CODEGEN_SAMPLE_RE``) once the on-disk migration is complete.
#
# NOTE: the legacy embedding names ``vle_*_clip`` / ``vle_*_siglip`` are *not*
# here — they pin a distinct CLIP/SigLIP backbone that remapping would collapse,
# silently mis-routing SigLIP data into CLIP cells. They live in the paper-figure
# registry's ``_LEGACY_PINNED_METHODS`` instead.
LEGACY_NAME_REMAP: dict[str, str] = {
    "llm_direct_qv_optimal":               "direct_qv",
    "llm_direct_qv_optimal_verif3":        "direct-verif16_qv",
    "llm_direct_qv_optimal_batch4_packed": "direct-batch4-packed_qv",
    "llm_direct_qv_optimal_batch8_seq":    "direct-batch8-seq_qv",
    "llm_direct_gvl_qv_optimal":           "gvl_qv",
    "llm_verifier_qv_optimal":             "verifier_qv",
    "llm_eureka_qv_optimal":               "eureka_qv",
    "llm_codegen_qv_optimal":              "codegen_qv",
    "llm_codegen_qv_optimal_mean":         "codegen-mean_qv",
    "llm_ranking_optimal":                 "ranking_qv",
    "delta_belief_ranking_optimal":        "delta-belief_qv",
    "sdpo_ranking_optimal":                "sdpo_qv",

    "llm_direct_sv_optimal":               "direct_sv",
    "llm_direct_sv_optimal_verif3":        "direct-verif16_sv",
    "llm_direct_sv_optimal_batch4_packed": "direct-batch4-packed_sv",
    "llm_direct_sv_optimal_batch8_seq":    "direct-batch8-seq_sv",
    "llm_direct_gvl_sv_optimal":           "gvl_sv",
    "llm_eureka_sv_optimal":               "eureka_sv",
    "llm_codegen_sv_optimal":              "codegen_sv",
    "llm_codegen_sv_optimal_mean":         "codegen-mean_sv",

    "vip_q":                               "vip_qv",
    "vip_v":                               "vip_sv",
    "liv_img_q_cos":                       "liv-cos_qv",
    "liv_img_v_cos":                       "liv-cos_sv",
    "liv_img_q_l2":                        "liv-l2_qv",
    "liv_img_v_l2":                        "liv-l2_sv",
    "liv_txt_q":                           "liv-txt_qv",
    "liv_txt_v":                           "liv-txt_sv",
}


# Legacy embedding names that pin a backbone via the ``method_name`` suffix
# (``_clip`` / ``_siglip``) instead of the EVAL dir's actor slot. Kept distinct
# from LEGACY_NAME_REMAP so the CLIP/SigLIP backbones are preserved rather than
# collapsed. Maps the legacy name -> (canonical base, pinned backbone actor);
# the paper-figure registry resolves the canonical base against the catalog and
# applies the backbone's display as the pinned model.
LEGACY_PINNED_EMBEDDING: dict[str, tuple[str, str]] = {
    "vle_gbr_clip":      ("vle-gbr", "clip"),
    "vle_gbr_siglip":    ("vle-gbr", "siglip"),
    "vle_cosine_clip":   ("vle-cosine", "clip"),
    "vle_cosine_siglip": ("vle-cosine", "siglip"),
    "vle_thresh_clip":   ("vle-thresh", "clip"),
    "vle_thresh_siglip": ("vle-thresh", "siglip"),
    "vle_softmax_clip":  ("vle-softmax", "clip"),
    "vle_softmax_siglip":("vle-softmax", "siglip"),
}


# ``method_name``s matching this regex are aggregated into a single "codegen"
# row. Matches both canonical (``codegen-s<N>_<qv|sv>``) and legacy
# (``llm_codegen_<qv|sv>_optimal_<s?N>``) sample names. Drop the legacy half
# once migration is complete.
CODEGEN_SAMPLE_RE = re.compile(
    r"^(?:codegen-s\d+_(?:qv|sv)|llm_codegen_(?:qv|sv)_optimal_s?\d+)$"
)
CODEGEN_AGG_DISPLAY = "codegen"
CODEGEN_AGG_FAMILY = "Code"
