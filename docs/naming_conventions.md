# qval prediction and evaluation naming conventions

> **Source of truth:** these conventions are implemented in
> `src/qval/registry/` (atoms + `naming.py`), and the actor / GT-actor /
> modality vocabularies and the method matrix are *derived* from the active
> catalog's `catalog.py` (the bundled one is `catalogs/qval_benchmark/catalog.py`).
> This document is the human-readable companion; when the two disagree, the
> registry wins. See `docs/registry.md`.

Scope: standardized prediction artifacts under a catalog's `data/predictions/`
and evaluation summaries under its `data/evaluations/` — the bundled catalog
roots these at `catalogs/qval_benchmark/data/`, and any other catalog mirrors the
same structure under its own root. Datasets (`shared/data/datasets/`) are out of
scope.

## 1. Top-level layout

```
catalogs/qval_benchmark/data/predictions/<env>/
├── GT_<num_points>_<dataset_id>_<actor>/
│   ├── gt/
│   │   └── <gt-method>_<ts>.json
│   ├── rollouts/
│   ├── predict.logs_<ts>.jsonl
│   └── ...
└── EVAL_<num_points>_<dataset_id>_<actor>_<modality>/
    ├── eval/
    │   ├── <method>[-<variant>]_<qv|sv>_<ts>.json
    │   └── <method>[-<variant>]_<qv|sv>_<ts>.py     (codegen samples only)
    ├── predict.logs_<ts>.jsonl
    └── ...
```

`<env>` ∈ `{terminal_bench, alfworld, frozen_lake, open_apps}`. The duplicate `<env>_vision/` top-level dirs are dropped — modality lives in the EVAL dir name.

`GT_` and `EVAL_` are siblings under `<env>/`. They share the same `<num_points>_<dataset_id>` prefix; their `<actor>` tokens may differ (e.g., GT actor `ds32` paired with EVAL actor `q35-122-or`). The eval YAML config explicitly pairs them via `prediction_sources`.

## 2. Directory tokens (canonical order)

| Slot            | Examples                                       | Notes                                              |
|-----------------|------------------------------------------------|----------------------------------------------------|
| `<num_points>`  | `100pt`, `200pt`                               | integer count of evaluation points + literal `pt` suffix |
| `<dataset_id>`  | `all-types-40ms`, `8x8`, `tblite-easy-40ms`, `eps025-ms45-recovery-som-filtered-zeros-ranking` | env-specific dataset identifier |
| `<actor>`       | see actor table below                          | who/what produced the predictions/rollouts in this dir |
| `<modality>`    | `text`, `vision`                               | EVAL dirs only; absent from GT dirs                |

### Actor table

| Token         | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `g4-26-or`    | Gemma4 26B-A4B (LLM)                                              |
| `g4-31-or`    | Gemma4 31B (LLM)                                                  |
| `q35-9-or`    | Qwen3.5 9B (LLM)                                                  |
| `q35-27-or`   | Qwen3.5 27B (LLM)                                                 |
| `q35-35-or`   | Qwen3.5 35B-A3B (LLM)                                             |
| `q35-122-or`  | Qwen3.5 122B-A10B (LLM)                                           |
| `ds32`        | DeepSeek v3.2 (strong agent baseline; commonly the GT-rollout actor) |
| `codex-55`    | Codex (strong agent baseline)                                     |
| `opus-47`     | Claude Opus 4.7 (strong agent baseline)                           |
| `random`      | Random policy                                                     |
| `scripted`    | Scripted/built-in policy                                          |
| `clip`        | CLIP backbone (used only by `vle-*` methods)                      |
| `siglip`      | SigLIP backbone (used only by `vle-*` methods)                    |
| `self`        | Pinned non-LLM Pre-trained methods (`vip`, `liv-*`)               |

## 3. Evaluation output layout

`scripts/pipeline/evaluate.py` (discovery mode) writes one evaluation directory
per environment/dataset/GT actor/eval actor/modality combination:

```
catalogs/qval_benchmark/data/evaluations/<env>/<dataset_id>_<gt_actor>_<eval_actor>_<modality>/
└── summary_<YYYYMMDD_HHMMSS>.json
```

The GT actor is always present, even when an environment currently has only one
GT set. This is load-bearing: plotting uses it to distinguish ordinary results
from GT-set ablations such as TerminalBench `codex-55` vs. `opus-47` vs.
`ds32`.

Examples:

```
catalogs/qval_benchmark/data/evaluations/terminal_bench/
├── tblite-easy-40ms_codex-55_g4-26-or_text/
│   └── summary_20260504_201001.json
├── tblite-easy-40ms_opus-47_g4-26-or_text/
│   └── summary_20260504_201026.json
└── tblite-easy-40ms_ds32_g4-26-or_text/
    └── summary_20260504_201052.json
```

Evaluation consumers treat old ambiguous directories without `<gt_actor>` (for
example `tblite-easy-40ms_g4-26-or_text/`) as non-canonical and ignore them.
When several `summary_<timestamp>.json` files exist in the same canonical
evaluation directory, consumers select the latest timestamp.

Figure/table consumers also filter to the current primary dataset id per
environment. For OpenApps, the primary dataset id is
`eps025-ms45-recovery-som-filtered-zeros-ranking`, corresponding to
`shared/data/datasets/open_apps/scripted_40traj_eps025_ms45_recovery_som_100pt_filtered_zeros_ranking_20260504_193708.pkl`.

## 4. File naming

### Inside `gt/`

```
<gt-method>_<ts>.json
```

GT methods used today:

- `qv_mc_max` / `qv_mc_mean` — Q-value MC ground truth
- `sv_mc_max` / `sv_mc_mean` — V-value MC ground truth
- `ranking_gt_qv_mc_max` / `ranking_gt_qv_mc_mean` — ranking-style GT derived from Q-value GTs
- `ranking_gt_sv_mc_max` / `ranking_gt_sv_mc_mean` — ranking-style GT derived from state-value GTs

### Inside `eval/`

```
<method>[-<variant>]_<qv|sv>_<ts>.json[+ .py for codegen]
```

The `_qv` / `_sv` token always appears even for ranking-style methods (they only ever use `_qv`). The `.py` sidecar exists for `codegen-s<N>` samples (one per sample).

## 5. JSON `method_name` field — load-bearing

**The `method_name` field inside each prediction JSON must equal the file basename minus the timestamp suffix.** This is what `evaluate.py` reads and writes into `summary.json`'s `estimations.<gt>.methods.<key>` keys, and what plotting reads via the registry. Renaming the file alone does *not* propagate; you must also patch the JSON.

Example: file `direct-batch8-seq_qv_20260427_102832.json` → JSON `{"method_name": "direct-batch8-seq_qv", ...}`.

## 6. Method + variant reference (rename map)

Apply this mapping when patching existing JSONs. **Old name = current `method_name` in JSONs and current filenames; New name = both new `method_name` and new file basename (sans `_<ts>`).**

| Old `method_name`                          | New `method_name`             | Display              | Family            |
|--------------------------------------------|-------------------------------|----------------------|-------------------|
| `llm_direct_qv_optimal`                    | `direct_qv`                   | direct-single        | Direct            |
| `llm_direct_sv_optimal`                    | `direct_sv`                   | direct-single        | Direct            |
| `llm_direct_qv_optimal_verif3`             | `direct-verif16_qv`           | direct-16            | Direct            |
| `llm_direct_sv_optimal_verif3`             | `direct-verif16_sv`           | direct-16            | Direct            |
| `llm_direct_qv_optimal_batch4_packed`      | `direct-batch4-packed_qv`     | direct-batched       | Direct            |
| `llm_direct_sv_optimal_batch4_packed`      | `direct-batch4-packed_sv`     | direct-batched       | Direct            |
| `llm_direct_qv_optimal_batch8_seq`         | `direct-batch8-seq_qv`        | direct-sequential    | Direct            |
| `llm_direct_sv_optimal_batch8_seq`         | `direct-batch8-seq_sv`        | direct-sequential    | Direct            |
| `llm_direct_gvl_qv_optimal`                | `gvl_qv`                      | gvl                  | Direct            |
| `llm_direct_gvl_sv_optimal`                | `gvl_sv`                      | gvl                  | Direct            |
| `llm_verifier_qv_optimal`                  | `verifier_qv`                 | verifier             | Intrinsic         |
| `llm_eureka_qv_optimal`                    | `eureka_qv`                   | eureka               | Code              |
| `llm_eureka_sv_optimal`                    | `eureka_sv`                   | eureka               | Code              |
| `llm_codegen_qv_optimal`                   | `codegen_qv`                  | codegen              | Code              |
| `llm_codegen_sv_optimal`                   | `codegen_sv`                  | codegen              | Code              |
| `llm_codegen_qv_optimal_mean`              | `codegen-mean_qv`             | codegen-avg          | Code              |
| `llm_codegen_sv_optimal_mean`              | `codegen-mean_sv`             | codegen-avg          | Code              |
| `llm_codegen_qv_optimal_s<N>` (N=0..15)    | `codegen-s<N>_qv`             | codegen *(aggregated)* | Code            |
| `llm_codegen_sv_optimal_s<N>` (N=0..15)    | `codegen-s<N>_sv`             | codegen *(aggregated)* | Code            |
| `llm_ranking_optimal`                      | `ranking_qv`                  | ranking              | Ranking           |
| `sdpo_ranking_optimal`                     | `sdpo_qv`                     | sdpo                 | Self-Distillation |
| `sdpo-gt_qv`                               | `sdpo-gt_qv`                  | sdpo-gt              | Self-Distillation |
| `delta_belief_ranking_optimal`             | `delta-belief_qv`             | $\Delta$belief       | Intrinsic         |
| `vip_q`                                    | `vip_qv`                      | vip                  | Pre-trained       |
| `vip_v`                                    | `vip_sv`                      | vip                  | Pre-trained       |
| `liv_img_q_cos`                            | `liv-cos_qv`                  | liv-cos              | Pre-trained       |
| `liv_img_v_cos`                            | `liv-cos_sv`                  | liv-cos              | Pre-trained       |
| `liv_img_q_l2`                             | `liv-l2_qv`                   | liv-l2               | Pre-trained       |
| `liv_img_v_l2`                             | `liv-l2_sv`                   | liv-l2               | Pre-trained       |
| `liv_txt_q`                                | `liv-txt_qv`                  | liv-txt              | Pre-trained       |
| `liv_txt_v`                                | `liv-txt_sv`                  | liv-txt              | Pre-trained       |
| `vle_gbr_clip`                             | `vle-gbr_sv`                  | vlm-rm               | Embedding         |
| `vle_gbr_siglip`                           | `vle-gbr_sv`                  | vlm-rm               | Embedding         |
| `vle_cosine_clip`                          | `vle-cosine_sv`               | vlm-rm-cos           | Embedding         |
| `vle_cosine_siglip`                        | `vle-cosine_sv`               | vlm-rm-cos           | Embedding         |
| `vle_thresh_clip`                          | `vle-thresh_sv`               | vlm-sor              | Embedding         |
| `vle_thresh_siglip`                        | `vle-thresh_sv`               | vlm-sor              | Embedding         |
| `vle_softmax_clip`                         | `vle-softmax_sv`              | vlm-sor-softmax      | Embedding         |
| `vle_softmax_siglip`                       | `vle-softmax_sv`              | vlm-sor-softmax      | Embedding         |

> Note on `vle-*`: clip and siglip variants get the **same** new `method_name` because the backbone now lives in the EVAL dir's `<actor>` slot (`..._clip_vision/`, `..._siglip_vision/`). The two are still distinguishable by their parent dir.

## 7. Worked examples

**LLM eval, ALFWorld, Qwen3.5 122B as eval actor, scripted as GT actor, 100 points of `all-types-40ms`, text:**

```
catalogs/qval_benchmark/data/predictions/alfworld/
├── GT_100pt_all-types-40ms_scripted/
│   ├── gt/
│   │   ├── qv_mc_max_20260427_102832.json
│   │   └── sv_mc_max_20260427_102832.json
│   ├── rollouts/
│   └── predict.logs_20260427_013108.jsonl
└── EVAL_100pt_all-types-40ms_q35-122-or_text/
    ├── eval/
    │   ├── direct_qv_20260427_102832.json
    │   ├── direct-batch4-packed_qv_20260427_102832.json
    │   ├── direct-batch8-seq_qv_20260427_102832.json
    │   ├── direct-verif16_qv_20260427_102832.json
    │   ├── gvl_qv_20260427_102832.json
    │   ├── eureka_qv_20260427_102832.json
    │   ├── verifier_qv_20260427_102832.json
    │   ├── codegen-mean_qv_20260427_102832.json
    │   ├── codegen-s0_qv_20260427_102832.json
    │   ├── codegen-s0_qv_20260427_102832.py
    │   ├── ... (s1..s15)
    │   └── ranking_qv_20260427_102832.json
    └── predict.logs_20260427_013108.jsonl
```

**Embedding methods, ALFWorld vision, CLIP backbone, 200 points of `all-types-40ms`:**

```
catalogs/qval_benchmark/data/predictions/alfworld/EVAL_200pt_all-types-40ms_clip_vision/eval/
├── vle-gbr_sv_20260427_102832.json
├── vle-cosine_sv_20260427_102832.json
├── vle-thresh_sv_20260427_102832.json
└── vle-softmax_sv_20260427_102832.json
```

**Pre-trained methods, ALFWorld vision, 200 points of `all-types-40ms`:**

```
catalogs/qval_benchmark/data/predictions/alfworld/EVAL_200pt_all-types-40ms_self_vision/eval/
├── vip_qv_20260427_102832.json
├── liv-cos_qv_20260427_102832.json
├── liv-l2_qv_20260427_102832.json
└── liv-txt_qv_20260427_102832.json
```
