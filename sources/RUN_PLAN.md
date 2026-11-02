# Loss attribution: extending to all arms and both architectures

The preliminary analysis ([`preliminary_findings.md`](preliminary_findings.md)) compared **llama base vs register α15** only. The old token losses contain just the arms `base` and `register15`. This plan applies every analysis to **all 4 interventions (register/wiki × α15/α35) in both architectures (llama_1b, qwen2_1.7b)**. It also adds the checks needed to make the claims rigorous.

## What the inventory showed (step 0 has already run: `outputs/manifest/`)

1. **Open issue 1 (corpus identity) is resolved.** The mixes trained on these synthetic token counts per document (SmolLM2 tokenizer):

   | Arm | Tokens/doc | Synthetic shards used |
   |---|---:|---:|
   | base | 448 | 113 |
   | reg15 | 468 | 108 |
   | reg35 | 577 | 87 |
   | wiki15 | 450 | 112 |
   | wiki35 | 550 | 92 |

   The earlier "~320 vs ~420" figure was wrong. Each synthetic shard *i* paraphrases FineWeb shard *i* (ids match 100%).

2. **Open issue 2 (packing confound) is real, and it is confounded with α.**
   - Each FineWeb shard is essentially a single CommonCrawl dump: the streaming shuffle buffer was too small.
   - The human half of each mix starts right after its last synthetic shard, so arms whose synthetic docs are longer see **different crawl dumps** in their human 10B tokens.

   | Arm | Human shards | Shared with base | Dump distance from base (TVD) |
   |---|---|---:|---:|
   | base | 113–181 | 100% | 0 |
   | reg15 | 108–176 | 93% | 0.07 |
   | wiki15 | 112–180 | 99% | 0.02 |
   | reg35 | 87–155 | 62% | **0.38** |
   | wiki35 | 92–160 | 70% | **0.31** |

   For example, reg35 and wiki35 contain no CC-2014-49, 2015-48 or 2021-43 human data, and about twice as much 2019-30, 2021-39 and 2021-49.
   - **Consequence:** with these 5 mixes, *α*, *synthetic doc length* and *human-half composition* are perfectly collinear (same rank order). Any a35 − a15 difference, such as the worse FineWeb loss for α35 arms, could come from the human half. Steps 4.4 and 4.5 measure how different the human halves are. Only a retrained control can fully separate the factors (see "Needs training").

3. **`final_eval.json` was wrong for every model; fixed on 2026-10-31.** Every domain held the FineWeb loss. The correct per-domain losses are in `trained_models/<model>/<run>/final_eval_fixed.json`, and the comparison is in [`../model_results_analysis/ppl/FINDINGS.md`](../model_results_analysis/ppl/FINDINGS.md).
   - **Wiki beats register on language modelling**, the reverse of the benchmark ranking.
   - **α35 is worse than α15 on language modelling,** and worse than base on FineWeb.

## What needs to be run

| # | Job | What it answers | Compute |
|---|---|---|---|
| 0 | `00_manifest.sbatch` ✅ done | data identity, dump composition, symlinked `corpora/` with the shards each mix really used plus a held-out reference (FineWeb shards 182–344) | 5 min |
| 1 | `01_token_losses.sbatch` | per-token NLL, entropy, top-1 prob and hit for **all 10 models on identical tokens** (7 domains × 1,024 sequences × 2,048). Uses an extended `compute_token_losses.py` (AutoModel for Qwen, adds the calibration stats) | 1 GPU, ~1–2 h |
| 2 | `02_attribution.sbatch` (array 0–5) | the **original** bucket attribution (token class, frequency, difficulty, position) for each arm vs its family's base: all tokens, and composition-controlled curves for docs ≥512 and ≥1024. Reproduces findings 1–5 for every arm | CPU in container, ≤1 h each |
| 3 | `03_cross_arm.sbatch` | new `cross_arm_attribution.py` (details below) | CPU, <1 h |
| 4 | `04_corpus_measures.sbatch` (array 0–5) | findings 6–8 for all 5 corpora, plus robustness and confound checks (details below) | 1 GPU each, ≤3 h |
| 5 | `05_synthesis.sbatch` | `dose_response.py` (details below) | 5 min |

**Step 3 (`cross_arm_attribution.py`) computes:**
- Overall Δ per domain, plus a macro over domains, with CIs.
- Paired **contrasts** between interventions on the same tokens: register − wiki at each α, and α35 − α15 for each source.
- **Profile correlations:** whether the same token types and documents improve across arms. **llama_X vs qwen_X serves as a replication across architectures**, using disattenuated r with split-half reliabilities.
- **Calibration:** NLL − entropy and ECE per difficulty quintile. This tests the leading hypothesis directly.
- A breakdown of the gain by whitespace token id, and the `<eos>` Δ per arm.

**Step 4 tasks:**
- **0:** token frequency (finding 6).
- **1:** distinct rare types (finding 7).
- **2:** predictability under Qwen2.5-1.5B (finding 8).
- **3:** predictability under **Pythia-1.4B**, to check that "less predictable" isn't just "less LLM-like".
- **4–5:** predictability and frequency of each arm's **human half** (the packing-confound check).

**Step 5 (`dose_response.py`):**
- Builds a per-arm table joining model-side gains (macro Δ, hard-tail Δ, rare enrichment, whitespace share, eos) with corpus-side measures (reference-NLL shift under both scorers, rare rate, rare types, doc length, human-half shift).
- Computes Spearman correlations per family (n = 4) and pooled (n = 8). These are descriptive.

**Submit everything with dependencies:** `bash scripts/submit_all.sh`.
- Step 0 runs again (idempotent, about 5 min). To skip it, submit 01 and 04 directly.
- Outputs land in `outputs/<step>/`. Each step has a `run.log` plus CSVs; steps 3 and 5 also write `SUMMARY.md` and `SYNTHESIS.md`.
- **Smoke test (2026-10-31):** all scripts ran end to end on 8 sequences per domain, with all 10 checkpoints loading.

## Questions this answers per arm

1. **Does the arm improve language modelling at all, and where?**
   - From the 8-sequence smoke test (not a result): α35 is roughly flat or negative on FineWeb and Gutenberg, and positive on arxiv and stack.
   - Step 1 at 1,024 sequences gives the real numbers.
2. **Is the gain still on hard and rare tokens with no cost on easy ones** (findings 2–3), and still not long-context (finding 1)? This is step 2.
3. **Is the whitespace/layout share similar across arms** (finding 4), and is the `<eos>` fingerprint stronger at α35 (finding 5)? Steps 2 and 3.
4. **Do all steered corpora have fewer rare occurrences, the same rare vocabulary, and lower predictability** (findings 6–8)? Does this hold under a non-LLM-era scorer? Step 4.
5. **Dose-response:** do arms whose corpus is *less predictable* gain more on the hard tail? Is the hard-tail gain carried by higher entropy (less over-confidence), as the calibration hypothesis predicts? Steps 3 and 5.
6. **Replication:** do llama and qwen trained on the same data improve the *same tokens*? If the disattenuated r is high, the token-level pattern is a property of the data, not of one training run. That is the best substitute for extra seeds available without training. Step 3.
7. **Confound:** are the α35 human halves measurably different in predictability or rare rate? If yes, a35 − a15 contrasts need that caveat, or the control run below. Steps 0 and 4.

## Follow-up steps (run 2026-10-31)

| # | Job | What it answers |
|---|---|---|
| 6 | `06_benchmark_losses.sbatch` | all 10 models on the exact lighteval prompts + gold answers (WikiTableQ, TREB-QA, SQuADv2, DROP, LAMBADA, TriviaQA), loss split into answer / table / passage / question / scaffold / demos |
| 7 | `07_corpus_format.sbatch` | format profile of the 5 synthetic corpora: Q/A layout, answer length, question types, tables, lists |
| 8 | `08_answer_content.sbatch` | answer content paired over identical source documents (shape, faithfulness to the source, vs benchmark gold answers) + benchmark answer gains by answer type → [`FINDINGS_answer_content.md`](FINDINGS_answer_content.md) |
| 9 | `09_answer_preference.sbatch` (submit with `--exclude=gn010`: its GPU fails CUDA init since 2026-11-01) | answer-form preference of every model after `Answer:` (span vs sentence, 0/3-shot) + classification of answer numbers not in the source → Findings 2, 4 and 5 of [`FINDINGS_answer_content.md`](FINDINGS_answer_content.md) |

Results: [`FINDINGS_benchmark_and_format.md`](FINDINGS_benchmark_and_format.md). The main findings for all arms are in [`FINDINGS_all_arms.md`](FINDINGS_all_arms.md).

## Needs training (cannot be done with existing models)

In order of value:

1. **Matched-human-half control.** Rebuild one mix (e.g. reg35) with `--human_start_shard 182` and do the same for base, so every arm's human half is identical. Retrain at least the α35 arm and base. This removes the confound in point 2 above.
2. **Temperature-matched baseline**, unchanged from the preliminary next steps. Calibrated on 2026-11-02 with the production script: T ≈ 1.08 matches register α35's predictability; the generation is set up but not started (see [`../temperature_baseline/README.md`](../temperature_baseline/README.md)). It is still the decisive test of "steering ≈ entropy injection".
3. **Extra seeds** for base and one steered arm per architecture.

## Lower priority, not scripted

- Overlap between `eval_fineweb` (FineWeb sample-10BT) and the training shards. The eval set stores only token ids, so this needs a hash-based check.
- A length-matched resampling of the synthetic corpora. It is mostly superseded now that lengths are known, and length is collinear with the human-half shift anyway.
