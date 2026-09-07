# Peer Review Report

**Manuscript:** Efficient Person Re-Identification via LoRA and SSF: A Comparative PEFT Study on ViT Backbone in TransReID

**Target Journal:** Expert Systems With Applications (ESWA)

**Review Date:** 20 July 2026

**Recommendation:** Major Revision

The manuscript is technically sound and carefully executed, but in its current form it is positioned as a computer-vision benchmarking study rather than an ESWA-style applied intelligent-systems contribution. Substantial revision is needed to align with the journal's scope, broaden the empirical base, and strengthen the practical/system-level contribution.

---

## 1. Summary of the Manuscript

The paper presents a controlled empirical comparison of two structurally distinct parameter-efficient fine-tuning (PEFT) methods, Low-Rank Adaptation (LoRA) and Scale-and-Shift Features (SSF), applied to the frozen Vision Transformer (ViT-Base) backbone of TransReID for person re-identification on Market-1501. Under a fixed 60-epoch training recipe, the authors vary depth placement (blocks 0–11, 4–11, 6–11), LoRA rank (r ∈ {8, 16, 32}) and scaling (α ∈ {16, 32, 48, 64}), module targeting (attention-only vs. attention+MLP), and optimizer configuration for SSF (LoRA-matched vs. official SSF settings). The main findings are: (i) depth placement dominates both methods; (ii) LoRA on blocks 4–11 offers the best accuracy–memory compromise (~25–30% VRAM reduction, mAP within ~5–6 points of full fine-tuning); (iii) SSF is extremely parameter-efficient (~2.87% trainable parameters) but suffers a larger accuracy gap and depends more strongly on full block coverage; (iv) parameter efficiency and memory efficiency are not interchangeable, since VRAM is governed by activation retention rather than trainable-parameter count. The paper concludes with a non-dominated frontier analysis and a table of practical configuration guidelines.

---

## 2. Overall Assessment

This is a clean, honest, and well-organized empirical study. The experimental control (fixed data pipeline, fixed seeds, matched training budgets, explicit optimizer-confound handling for SSF) exceeds what is typical for PEFT comparison papers, and the reporting of peak VRAM alongside accuracy is genuinely useful. However, the contribution is a comparative benchmark of existing methods on one backbone and one dataset, with no methodological novelty and no deployed system component. For ESWA specifically, whose remit emphasizes intelligent systems with demonstrable applied impact and methodological contribution, the current framing sits at the boundary of scope, and the empirical base is too narrow to support the generality of the design guidelines that constitute the paper's main claimed deliverable.

---

## 3. Strengths

**S1. Experimental rigor and control.** A single fixed training recipe across all runs, consistent seeds, an identical data pipeline for both methods, and a two-case optimizer design that explicitly surfaces the optimizer confound for SSF. This last point is a genuine methodological strength — rarely seen in PEFT comparisons.

**S2. Practically meaningful metrics.** Reporting peak GPU memory, trainable-parameter ratio, and wall-clock time alongside mAP/CMC directly addresses the deployment question practitioners actually face. The observation that memory is dominated by activation retention, not parameter count, is well argued and correctly explained.

**S3. Novel application pairing.** To the reviewer's knowledge the claim that no prior work systematically compares multiple PEFT methods on a ViT-based Re-ID backbone is plausible; the gap identification in Section 2.8 is clearly articulated.

**S4. Actionable synthesis.** The non-dominated frontier analysis (Eq. 3) and Table 3's constraint-to-configuration mapping convert raw results into usable guidance, which is exactly the kind of deliverable an applied journal values.

**S5. Honest limitations section.** The authors candidly scope the study (single backbone, single dataset for SSF, bounded grids) rather than over-claiming, and the failure case (r=16, α=64 collapse) is reported rather than hidden.

**S6. Reproducibility.** Public code repositories for both experimental phases are provided.

---

## 4. Weaknesses

### 4.1 Major weaknesses

**W1. Scope fit with ESWA.** The manuscript is written as a computer-vision benchmarking paper. ESWA expects a clear contribution to intelligent/expert systems with applied impact; a comparative study of two existing methods, without a new method, framework, or system, will be questioned on scope. The 'configuration-effect map' and design guidelines are the strongest candidates for an ESWA-style contribution, but they are currently presented as a by-product rather than as a decision-support methodology.

**W2. Single-dataset evaluation.** All conclusions rest on Market-1501, which is small, dated, and known to saturate. The paper's central deliverable, generalizable design guidelines, cannot be validated from one dataset. At minimum, MSMT17 (larger, more domain shift) or DukeMTMC-style cross-domain evaluation is needed to show that the depth-placement and rank–scaling patterns transfer. The authors themselves acknowledge this for SSF, but it applies equally to LoRA.

**W3. Single run per configuration.** Seeds are 'kept consistent', implying one run per cell. Several reported gaps (e.g., 0.2–0.5 mAP differences between SSF Case 1 and Case 2; 74.8 vs. 75.3 mAP among 6–11 LoRA settings) are almost certainly within seed-to-seed variance for Market-1501. Without mean±std over ≥3 seeds, claims such as 'Case 2 consistently achieves marginally better results' are not statistically supportable, and the frontier membership of near-tied configurations is fragile.

**W4. Accuracy gaps are large and under-analyzed.** The best partial-coverage LoRA is still 4.8 mAP below baseline, and 0–11 LoRA at r=8 loses 2.2 mAP while saving almost no memory (11.4 vs. 11.5 GB). A skeptical reader may conclude that, on this benchmark, PEFT is simply not yet compelling for Re-ID. The paper needs either stronger configurations (e.g., longer schedules for PEFT runs, LoRA on LayerNorm, DoRA/AdaLoRA variants) or a deeper analysis of *why* the gap persists (feature-space analysis, per-class breakdown, retrieval error modes).

**W5. Missing comparison breadth.** Only two PEFT methods are compared, yet the title and abstract promise a 'comparative PEFT study'. Adapters, VPT, BitFit, and LN-tuning are standard baselines in the ViT-PEFT literature cited by the authors ([1], [14], [16]) and are cheap to run. Their absence weakens both the comparative claim and the practical guidelines (a practitioner cannot know whether LoRA beats a plain adapter here).

**W6. The strong r=8 vs. weak r=16 anomaly at 0–11 needs investigation.** A 10.3 mAP drop from r=8 to r=16 at fixed α=16 (85.8 → 75.5) is unusually severe and is explained only by a one-line over-perturbation argument. With α/r scaling, r=16/α=16 actually implies a *smaller* update multiplier than r=8/α=16, which contradicts the stated explanation. This inconsistency must be resolved (learning-rate coupling? optimizer state? an implementation issue?) because it underpins the 'higher rank is detrimental at full depth' guideline.

### 4.2 Minor weaknesses

**W7.** Section 5 cites 'Section IV' (Roman numeral, IEEE style) while the manuscript uses Arabic numbering; formatting is inconsistent with Elsevier style in several places.

**W8.** Reference [2] ('Iclr 1 (2) (2022) 3') is malformed; [16] and [18] use 'et al.' in author lists, which Elsevier style does not permit; [11] lacks page/article numbers.

**W9.** Wall-clock training time is listed as a recorded measurement (Section 3.3) but never reported in any table.

**W10.** The batch size is described only as 'fitting the most memory-intensive configuration'; the actual value is never stated, harming reproducibility despite released code.

**W11.** Rank-10 is dropped from Table 2 without explanation while present in Table 1.

**W12.** The ethical considerations paragraph (5.3) is appropriate but generic; ESWA increasingly expects concrete discussion (e.g., GDPR-style constraints on Re-ID deployment, dataset consent status of Market-1501).

**W13.** Figure 1 is informative but very dense; sub-captions repeat body text nearly verbatim.

---

## 5. Suggestions for Improvement

To address scope fit (W1), the single most impactful revision: reframe the contribution as a decision-support methodology for resource-constrained model adaptation. Concretely: formalize the configuration-effect map as an explicit mapping from deployment constraints (VRAM budget, parameter budget, accuracy floor) to recommended configurations; present the non-dominated frontier as the core of a recommendation procedure (even a simple lookup/rule system); and validate that the procedure's recommendations transfer to a second dataset. This converts a benchmark into an expert-system-style contribution that matches ESWA's remit without requiring new algorithms.

**R1** (addresses W2). Add at least one additional Re-ID benchmark (MSMT17 strongly preferred) for both LoRA and SSF, at least for the frontier configurations. Even a reduced grid (best/worst/middle configurations per regime) would substantiate generality.

**R2** (addresses W3). Re-run frontier and near-tied configurations with 3 seeds; report mean ± std and mark differences smaller than the observed std as ties. Soften or remove claims currently resting on ≤0.5 mAP margins.

**R3** (addresses W5). Add two cheap baselines: BitFit (bias-only tuning) and classical bottleneck adapters. Both are a few lines of code on a frozen backbone and would materially strengthen the comparative claim.

**R4** (addresses W6). Diagnose the 0–11 r=16 collapse: sweep learning rate for that cell, verify the α/r implementation, and report training curves. Either explain the mechanism convincingly or correct the result.

**R5** (addresses W4). Explore whether the LoRA gap closes with a longer schedule (PEFT methods often need more epochs than full FT at matched LR), or with LoRA additionally on LayerNorm/patch-embedding. If the gap is fundamental, add analysis (e.g., CKA similarity of adapted vs. fully fine-tuned features) explaining where PEFT falls short for retrieval-style objectives.

**R6.** Report wall-clock time and throughput (it is promised in 3.3), the batch size, and total GPU-hours; these directly support the deployment narrative.

**R7.** Add a cost model or at least a plot: mAP vs. peak VRAM with the frontier drawn explicitly. Currently the frontier exists only as prose; a figure would make the central contribution legible at a glance.

**R8.** Fix reference formatting throughout (W7, W8), unify section cross-references, state Rank-10 for Table 2 or justify its omission, and tighten Figure 1 captions.

**R9.** Strengthen Section 5.3 with concrete governance measures and a statement on dataset licensing/consent status; consider citing recent Re-ID ethics literature.

**R10.** In the abstract and introduction, lead with the decision-support framing (constraint → configuration mapping) rather than the benchmark framing, and state explicitly what an ESWA reader can *use* from the paper.

---

## 6. Questions for the Authors

**Q1.** What batch size and learning-rate schedule per parameter group were used, and was the learning rate re-tuned per PEFT method or inherited from the full fine-tuning recipe? An inherited LR could disadvantage PEFT and inflate the reported accuracy gaps.

**Q2.** How do you reconcile the claim that higher rank 'over-perturbs' at 0–11 with the α/r update scaling, under which r=16/α=16 yields a smaller multiplier than r=8/α=16?

**Q3.** Were SSF modules initialized at identity (γ=1, β=0)? If not, this could explain SSF's sensitivity to partial coverage.

**Q4.** Is the full fine-tuning baseline (88.0 mAP) obtained with the identical 60-epoch recipe, and how does it compare to the published TransReID result under the same augmentations?

**Q5.** For the 4–11 regime, why does peak VRAM *decrease* from r=16/α=48 (8.01 GB) to r=32/α=64 (7.84 GB) despite more trainable parameters? Measurement noise, or a batch/allocator effect worth explaining?

---

## 7. Detailed Verdict by Criterion

| Criterion | Assessment |
|---|---|
| Novelty / Originality | Moderate: first PEFT comparison on ViT Re-ID; no methodological novelty. |
| Technical soundness | Good, with two open issues (W3 single-seed; W6 rank anomaly). |
| Experimental rigor | Above average for the genre; controls are a clear strength. |
| Significance / Impact | Currently limited by single dataset and two-method scope. |
| Fit to ESWA scope | Borderline as written; achievable with decision-support reframing. |
| Clarity / Presentation | Good; minor formatting and reference issues. |
| Reproducibility | Good (public code), missing batch size and timing details. |

---

## 8. Final Recommendation

**Major Revision.** The underlying experimental work is solid and the deployment-oriented measurements are valuable. The manuscript is not acceptable for ESWA in its current form for three reasons: (i) the framing is that of a CV benchmark rather than an applied intelligent-systems contribution; (ii) the empirical base (one dataset, one seed, two methods) is too narrow to support the generality of the guidelines that constitute the paper's main deliverable; and (iii) one central result (the full-depth rank anomaly) is internally inconsistent with the paper's own explanation. All three are fixable. If the authors (a) reframe the configuration-effect map as a validated decision-support methodology, (b) add one dataset and multi-seed results for the frontier configurations, (c) add two lightweight PEFT baselines, and (d) resolve the rank anomaly, the paper would be a competitive ESWA submission. Alternatively, if the authors prefer to publish the study largely as-is, a venue such as Pattern Recognition Letters, Image and Vision Computing, or Machine Vision and Applications would be a more natural fit for the current framing.