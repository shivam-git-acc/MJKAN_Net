# Agentic AI Usage

## Overview

MJKAN-Net was developed through an intensive human–AI collaboration using **Claude** (Anthropic) as the primary development partner across the full research lifecycle: problem framing, architecture design, the 17-model ablation ladder, rigorous evaluation, and repository packaging. The collaboration was **agentic in the research sense** — not just code autocompletion, but an iterative loop of *hypothesize → instrument → verify → reframe*, where the AI proposed experiments, wrote self-contained Kaggle cells, computed results, and (critically) pushed back when a number looked too good to be true.

The single most important pattern of this project was **verification over optimism**: again and again, a promising result was interrogated rather than accepted, and several headline numbers were *discarded* because they turned out to be artifacts. That discipline — described honestly in the "What Did Not Work" section — is what makes the final results defensible.

---

## Agentic Workflows Used

### 1. Reasoning & Planning

Before writing code, the AI was used to design the overall **ablation ladder** so that every capability of the flagship would be attributable to a specific, ablated decision. Key design decisions made through AI-assisted planning:

- **Framing the problem as contrastive representation learning** rather than plain classification — so that classification happens by k-NN in `z`-space, which is what makes the system extensible to unseen traffic without retraining, and what enables the novelty-detection capability.
- **Deploying on `z` (projection output), not `h` (encoder output)** — decided from evidence: across experiments `z` gave intra ≈ 0.96 / inter ≈ 0.35 versus `h`'s 0.72 / 0.74. The AI surfaced this gap and the decision followed the data.
- **GatedFiLM over vanilla FiLM** — the gate prevents degenerate conditioning on flows where context is uninformative.
- **Using a 6-feature *full-flow* context vector** deliberately chosen to be information the SSM cannot recover from the 30-packet window — with collector-specific fields (`FLOW_ENDREASON`) explicitly excluded as non-transferable.
- **Structuring the V-series ladder** so each rung isolates one hypothesis (joint exposure → conditioning → adversarial alignment → separation loss → condition invariance).

### 2. Coding Assistant — Model Implementation

Each model variant was implemented with the AI suggesting architecture details and catching subtle errors that would have silently corrupted results:

- **Pure-PyTorch SSM** with correct mask handling for variable-length sequences (the `mask.view(B,L,1,1)` trick to zero padding inside the recurrence) — avoiding the common bug of letting padded positions leak into the state.
- **GatedFiLM zero-initialization** of the final modulation layer, preventing random conditioning at the start of training.
- **SupCon numerical stability** — the `sim − sim.max().detach()` shift before exponentiation.
- **The BatchNorm projection-head fix.** Early SupCon runs stalled (loss plateaued ~5.4). The AI diagnosed this by monitoring `neg_sim`, traced it to a plain projection head + too-low LR + cosine-to-zero schedule, and fixed it with a BatchNorm head (`Linear→BN→ReLU→Linear`) + warmup-then-constant LR at 1e-3. This single fix unblocked the entire contrastive line of work.
- **WeightedRandomSampler** for the ecommerce / cdn_web_assets class imbalance.

### 3. Tool Chaining — Evaluation & Data Pipelines

The most agentic part of the project was the **evaluation infrastructure**, built as multi-step pipelines that read from HuggingFace, recomputed metrics under one consistent method, and gated every result behind a sanity check:

1. **Master scorecard** — a pipeline that downloaded every checkpoint, auto-detected which of three architecture variants it was (by inspecting the state-dict keys), loaded it with `strict=True`, and recomputed all KPIs under one uniform evaluation method so the 17 models were directly comparable.
2. **Unseen-app benchmark builder** — extracted genuinely-novel apps from both QUIC and TLS raw data, matched the exact CESNET feature recipe, stored raw features for reproducibility, and pushed the benchmark to HuggingFace (`generalization/unknown_apps_combined.npz`).
3. **Novelty-detection evaluation** — computed threshold-free AUROC and an operating-point trade-off table, replacing an earlier hand-picked threshold.
4. **Cross-pipeline test** — parsed a raw 515 MB pcap with scapy into flows and features, with a hard **scale-sanity gate** that flagged the two CESNET-exporter-specific features as unreconstructable.

### 4. Memory & Context Handling

Across many sessions, persistent memory retained the project's evolving state: the HuggingFace folder layout, confirmed feature formulas, the exact normalization stats per checkpoint, the list of which numbers were verified versus pending, and the standing methodological principles (e.g. "always load `strict=True`", "verify scale before trusting any cross-distribution number"). Between sessions the conversation was summarized automatically, and the memory index kept the path map and pending-task list available at the start of new work — which mattered for a project with 17 models, two datasets, and dozens of checkpoints.

---

## What Worked Well

### The hypothesize → instrument → verify → reframe loop
The highest-value pattern was treating every result as a hypothesis to be tested, not a fact to be reported. Examples that paid off:
- The V5z DANN result was *expected* to help, came back redundant, and was correctly reframed as an honest negative ("joint training already achieved invariance") rather than dropped.
- The "behavior-distinctive generalization" finding emerged only because per-app results were inspected instead of being collapsed into a single average — revealing that search/messaging/audio unseen apps classify at 90–99% while gaming/storage fail *predictably* by behavioral overlap.

### Scale-sanity gates before trusting any number
Every cross-distribution and novel-app evaluation was gated behind a check that normalized features must land at ≈ 0 mean / 1 std. This gate **caught two would-be-false findings**: the 5G-YouTube and raw-pcap tests both showed zero-variance context features, exposing them as pipeline artifacts before they could be written up as "distribution shift."

### Architecture auto-detection for the scorecard
Inspecting state-dict keys to identify which of three architectures a checkpoint used — and loading with `strict=True` — turned a fragile manual process into a robust one. `strict=True` was adopted specifically as an **honesty guard**: a mismatch now fails loudly instead of silently loading a partially-random model and producing a misleading accuracy.

### Self-contained Kaggle cells
Keeping each experiment as a standalone notebook cell (rather than a file-based module tree) matched the actual development environment and made every result independently reproducible — and made it trivial to re-run a single experiment from HuggingFace after a session reset.

### Reproducible benchmarks on HuggingFace
Storing the unseen-app benchmark as **raw** features (not pre-normalized) means it can be re-evaluated against any model by applying that model's own normalization — a deliberate choice that makes the generalization result re-runnable by a third party, which directly supports the reproducibility requirement.

---

## What Did Not Work

This section is deliberately detailed, because the failures are where the rigor lived.

### A headline accuracy number that was a leakage artifact
An early real-time model scored ≈ 0.95 on QUIC under a random train/test split. On inspection this was **near-duplicate leakage**: adjacent flows in a random split are nearly identical, so the test set was effectively memorized. Switching to a temporal split (train on earlier weeks, test on later weeks) dropped the honest number to ≈ 0.84. **We report only the temporal number.** The 0.95 was discarded — catching it required suspicion of a too-good result, not acceptance of it.

### A baseline that silently loaded at chance
The 6-class baseline scored 0.119 (near-random) even after a clean `strict=True` load. The cause was a **normalization mismatch**: the on-disk pool data was raw, but the checkpoint expected input normalized with its own stored stats. The prediction distribution collapsing entirely into one class was the tell. The fix (apply the checkpoint's own normalization) is correct, but the episode is the lesson: **a number can be wrong even when the model loads perfectly.** This is exactly why `strict=True` + a scale check became standing policy.

### Two cross-distribution tests that hit a structural wall
Validating on externally-captured YouTube traffic (5G feature set, and a raw pcap) **failed in a way that taught us something real.** Two context features (`roundtrip_density`, `truncation_ratio`) are CESNET-exporter-derived and *cannot* be reconstructed from raw packets — knowing the correct formula does not help, because the formula's inputs are themselves exporter outputs. The scale-sanity gate caught this (zero-variance features), so the misleading "YouTube classifies as search" result was correctly identified as a **pipeline-mismatch artifact, not distribution shift**, and written up honestly as a *deployment finding* rather than a *model finding*. The temptation to report the artifact as a distribution-shift result was explicitly resisted.

### The "66% flagged" novelty metric
An early novelty-detection figure relied on a hand-picked confidence threshold (`known_mean − 0.05`) with no false-positive accounting. It was retired in favor of **threshold-free AUROC (0.832)** plus an operating-point trade-off table — the honest, standard way to report a detector. The arbitrary 0.05 margin would not have survived scrutiny.

### HTTP transfer (Czech HTTPS) — exact-class generalization did not hold
We tested the model on a genuinely unseen protocol: a **Czech HTTPS** capture (different protocol, different country/network, different feature distribution). At the level the KPI cares about — *exact-class* accuracy — it **did not work**: convention-free mapped accuracy was only **V2-FiLM 28.2%, V7 27.6%, V5z 11.4%.** Two compounding causes, both diagnosed rather than hand-waved: (a) a real **feature-distribution shift** (the Czech `size` feature normalized to mean −1.149 vs CESNET's +0.006 — the same feature-comparability wall the pcap/5G tests hit), and (b) genuine **behavioral overlap** between video and bulk-download under encryption. The honest reading is that exact-class transfer to an unseen, distribution-shifted protocol is *not* something this model achieves — and we report the low numbers plainly. (The one positive that *did* survive — ≈ 95% of flows landing in the correct *behavioral family* — is documented in `architecture.md`, but it is explicitly a family-level result, not the class-level transfer the KPI would require. We were careful not to let the family-level success paper over the class-level failure.) Notably, the DANN variant (V5z) scored **worst** on this transfer (11.4%), consistent with the earlier finding that adversarial protocol-erasure adds nothing generative.

### Limits of automated detection across long sessions
For a multi-session project with 17 models, fine-grained code state (exact line numbers, which checkpoint had which normalization) could not always be carried perfectly across session boundaries; files had to be re-read at the start of new sessions. High-level decisions persisted via memory, but precise code state did not — so the practical workflow was: re-confirm the artifact's current state before editing, rather than trusting a remembered line number.

---

## Honest Reflection: What the AI Collaboration Actually Contributed

The AI's most valuable contribution was **not** writing code faster — it was acting as a **skeptical second scientist**. The defining results of this project (the verified 0.91 temporal accuracy, the behavior-distinctive generalization theory, the AUROC 0.832 novelty detection, the cross-pipeline-coupling deployment finding, and the honest Czech-HTTP transfer result) all exist because results were *interrogated* rather than accepted: the leakage was caught, the normalization bug was caught, the pipeline artifacts were caught, the arbitrary threshold was replaced, and the unseen-protocol transfer was reported with its low class-level numbers stated plainly rather than buried beneath its family-level success. A collaboration that simply maximized headline numbers would have reported 0.95 accuracy, "unseen apps classify as X" distribution-shift claims, "66% novelty detection," and a cherry-picked "95% HTTP transfer" — all of which we now know to be wrong or misleading. The honest, defensible submission is the direct product of the verify-don't-trust loop.

---

## Tools and Setup

| Tool | Role |
|------|------|
| Claude (Anthropic) | Primary research & development partner: planning, coding, evaluation, verification |
| HuggingFace Hub | Model, data, and benchmark hosting; `hf_hub_download` for reproducible access |
| Kaggle (T4 GPU) | Compute environment; all experiments as self-contained notebook cells |
| Weights & Biases | Training-metric tracking |
| scikit-learn | k-NN evaluation, AUROC, t-SNE |
| scapy | Raw-pcap parsing for the cross-pipeline test |
| python-dotenv | Secrets management |

No MCP servers or multi-agent orchestration were used. All work was a single human–AI collaboration, with the human directing the research, asking probing conceptual questions, correcting imprecise claims, and making final design decisions — and the AI proposing experiments, implementing them, computing results, and flagging when a result should not be trusted.
