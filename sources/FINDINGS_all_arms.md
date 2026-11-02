# Loss attribution across all arms and both architectures

This extends [`preliminary_findings.md`](preliminary_findings.md), which covered llama base vs register α15 only. Every analysis now covers the 4 interventions (register/wiki × α15/α35) in both architectures (llama_1b, qwen2_1.7b).

**Runs:** 2026-10-31, SLURM jobs 210276–210281; the setup is in [`RUN_PLAN.md`](RUN_PLAN.md).

**Token losses:** all 10 models were scored on **identical tokens**: 7 domains × 1,024 sequences × 2,048 tokens = 14.7M tokens per model.

**Notation:** Δ = loss(base) − loss(arm), in **centi-nats** (nats × 100) unless stated otherwise; positive = the steered model is better. CIs come from a bootstrap over eval sequences.

**Outputs:**

| Analysis | Location |
|---|---|
| Overall Δ, contrasts, calibration, profile correlations | `outputs/cross_arm/` (`SUMMARY.md`) |
| Bucket attribution | `outputs/attribution_<family>_{all,len512,len1024}/` |
| Corpus measures | `outputs/corpus/` |
| Per-arm synthesis | `outputs/synthesis/` (`SYNTHESIS.md`) |

---

## 1. The gains, and how they differ between arms

The per-domain Δ agrees with the full-split perplexity recomputation (`model_results_analysis/ppl/FINDINGS.md`):
- **wiki_a15 > wiki_a35 ≈ register_a15 > register_a35**, in both architectures.
- Wiki beats register mainly on arXiv (−1.0 to −3.0) and Wikipedia (−0.4 to −1.6); register wins only on HackerNews.
- α35 is worse than α15 overall.

**New: removing whitespace tokens shows where α35 actually loses** (Δ on non-whitespace tokens / Δ on whitespace tokens):

| Domain | α15 arms, words | α15 arms, whitespace | α35 arms, words | α35 arms, whitespace |
|---|---:|---:|---:|---:|
| arXiv | +3.3 to +4.4 | +2 to +16 | +1.8 to +5.3 | +9 to +17 |
| Stack | +1.9 to +2.5 | +9 to +11 | +1.2 to +2.0 | +12 to +14 |
| Wikipedia | +1.0 to +1.5 | +6 to +11 | +0.6 to +2.1 | +8 to +17 |
| HackerNews | +1.2 to +1.8 | −1 to +2 | +1.1 to +2.2 | 0 to +7 |
| FineWeb | +0.4 to +0.6 | −1 to 0 | **−0.2 to −0.4** | 0 to −2 |
| FreeLaw | +0.2 to +0.8 | +5 to +12 | **−0.5 to −1.1** | +12 to +17 |
| Gutenberg | +0.8 to +2.0 | −4 to +6 | **−0.8 to +0.2** | +3 to +12 |

- **At α35, the models predict word tokens *worse* than base on FineWeb, FreeLaw and Gutenberg** (web prose, legal prose, literature). This holds for both register and wiki, in both architectures.
- The small net gains α35 still shows there come **entirely from whitespace and layout tokens**, which improve *more* at α35. That is why the whitespace share of the gain exceeds 100% in those domains in `SUMMARY.md`.
- α15 arms improve words in every domain.

## 2. Where the gain sits (findings 1–5, now for every arm)

Values are medians over the 7 domains, from `attribution_<family>_all`:

| Bucket (mean Δ) | llama reg15 | llama reg35 | llama wiki15 | llama wiki35 | qwen reg15 | qwen reg35 | qwen wiki15 | qwen wiki35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easiest quintile | −0.02 | +0.06 | −0.03 | −0.04 | +0.04 | +0.03 | −0.10 | −0.12 |
| q2 | +0.40 | −0.04 | +0.27 | −0.27 | +0.21 | −0.50 | −0.11 | −0.08 |
| hardest quintile | +3.0 | +4.0 | +3.8 | +5.0 | +3.1 | +4.4 | +5.7 | +3.8 |
| unseen in base corpus | +9.4 | +9.0 | +10.5 | +11.6 | +5.5 | +2.2 | +7.9 | +7.9 |
| rank > 10k | +3.2 | +2.6 | +4.2 | +4.9 | +3.3 | +3.4 | +3.4 | +3.7 |
| top-100 tokens | +0.84 | +0.84 | +0.81 | +0.26 | +0.38 | +0.37 | +0.17 | +0.07 |
| **function words** | **+0.97** | **+0.04** | **+0.91** | **+0.17** | **+0.96** | **−0.11** | **+0.74** | **−0.21** |
| whitespace | +7.3 | +10.8 | +5.2 | +12.3 | +5.5 | +9.5 | +6.4 | +10.4 |

1. **Not long-context (finding 1 holds for all 8 arms).**
   - On composition-controlled curves (complete documents ≥ 512 or ≥ 1024 tokens), almost every significant slope is *negative*: the gain is front-loaded in FreeLaw, Wikipedia and Stack.
   - The only positive slopes are small: Stack for llama reg35 and wiki15, and HackerNews for qwen reg35 and wiki35.
2. **The gain sits on hard tokens with no cost on the easiest (finding 2 holds).** The hardest quintile carries 1.8–4.3× its share of the gain, and the easiest is about 0 for every arm. **New:** α35 arms give up the gain on *moderately easy* tokens (q2, and function words) while gaining more on the hardest quintile.
3. **The gain sits on rare tokens (finding 3 holds).**
   - Tokens unseen in the base corpus gain +2 to +12 centi-nats, and rank > 10k gain +2.6 to +4.9.
   - Top-100 tokens gain little, and at α35 almost nothing.
4. **Whitespace/layout carries a large share in FreeLaw, Stack and Wikipedia (finding 4 holds).** It grows with α: at α35 whitespace gains are about 1.5–2× those at α15.
5. **`<eos>` fingerprint (finding 5 holds, and is stronger at α35).** Every arm predicts end-of-document worse, e.g. qwen reg35 at −3 to −4 nats on Gutenberg and Stack. It covers < 0.1% of tokens and doesn't affect perplexity: excluding EOS changes the macro Δ by ≤ 0.1.

## 3. The training corpora (findings 6–8, all 5 corpora, only the shards actually used)

| Corpus vs base | Rare rate | Rare types (4M tokens) | Uses per rare type | Qwen2.5 ref-NLL Δ, common tokens | Pythia ref-NLL Δ, common tokens | Qwen Δ, rare first mention | Pythia Δ, rare first mention |
|---|---:|---:|---:|---:|---:|---:|---:|
| reg15 | 0.92× | 1.01 | 0.93× | −0.11 | −0.07 | −0.06 | +0.01 (ns) |
| wiki15 | 0.93× | 1.00 | 0.94× | −0.11 | −0.07 | −0.05 | 0.00 (ns) |
| reg35 | **0.82×** | 0.99 | **0.85×** | **−0.28** | **−0.21** | −0.16 | −0.05 |
| wiki35 | **0.85×** | 1.00 | **0.87×** | **−0.25** | **−0.19** | −0.12 | 0.00 (ns) |

The two ref-NLL columns are reference-model NLL in nats; negative = less predictable than base.

- **Findings 6–8 replicate for every corpus, with a clear α dose-response:**
  - fewer rare-token occurrences;
  - the same rare vocabulary;
  - less predictable text;
  - α35 corpora are roughly 2.5–3× further from base than α15.
- **Register and wiki corpora are almost indistinguishable** on all of these measures. So these measures can't explain why wiki beats register on perplexity, or why register beats wiki on benchmarks.
- **Scorer check (new).**
  - Under Pythia-1.4B (pre-LLM), the steered text is still clearly less predictable, so "less predictable" isn't just "less LLM-like".
  - But under Pythia, **first mentions of rare words are as predictable as in base**. The unpredictability sits in common tokens and repeats, meaning in phrasing, not in how rare words are introduced.

## 4. The leading hypothesis needs revising

**Hypothesis being tested:** "less predictable training text → better calibration on the hard tail".

- **Dose-response is not monotone, it's an inverted U.**
  - Across the 8 arms, the macro gain correlates *positively* with corpus predictability: Spearman ρ = 0.8 in each family for the Qwen ref-NLL Δ.
  - So arms with the *least* predictable corpora (α35) gain *least*. Base (fully predictable) < α15 (somewhat less predictable) > α35 (much less predictable).
  - A small dose of unpredictability helps; a large dose costs prediction of ordinary words in prose domains (section 1).
- **Hard-tail gains don't come from flatter predictions.**
  - On the hardest quintile, NLL drops by 2–5 centi-nats while the **predictive entropy barely changes** (Δ entropy −0.6 to +3.0, mostly ≈ 0). The base model's over-confidence gap there is huge (NLL − entropy ≈ 3.3 nats), so the gap shrinks.
  - That happens because the steered models put more probability on the *correct* rare continuation at the same overall uncertainty, not because they hedge more.
  - "Better calibration" in the sense of "less over-confident" is true, but only as a consequence of better ranking. The **"entropy injection" reading (flatter distributions) is not supported**.
- **The hard-tail gain doesn't track corpus unpredictability** (ρ = 0.4 under Qwen, ≈ 0 under Pythia). What changes with α is the *cost* side: function words, q2 tokens, and prose domains.

**Revised picture:**
- Paraphrasing with a steering vector diversifies phrasing (less formulaic text, fewer repeats of rare words), and that improves prediction of rare and hard tokens in human text.
- At α15 this comes at no cost. At α35 the phrasing drifts far enough from human usage that the model's prediction of ordinary function words and prose gets worse, while layout and whitespace modelling keeps improving.

## 5. Do the same tokens improve? (replication across architectures)

Per-token-type Δ profiles, pooled over domains, 20,202 types; disattenuated r:
- **Same condition across architectures (llama_X vs qwen_X): r = 0.52–0.62.**
- Different conditions: 0.34–0.54. For example, register vs wiki across architectures gives 0.34–0.44.
- So the arms share a common component, presumably generic "paraphrased-data" gains on rare and hard tokens. Each data condition also adds its own token-level signature, and **that signature replicates across architectures**: it is a property of the data, not of one training run.
- The split-half reliability of the profiles is 0.64–0.71, so the condition-specific part is real but not dominant.

## 6. Confounds

- **Human-half shift (α35 mixes see different FineWeb dumps).** It is real in composition: dump distance (TVD) is 0.31–0.38 for α35 vs ≤ 0.07 for α15. But it is small on the measures that matter:
  - the human halves' rare rate is +0.8% to +3.7% vs base;
  - their predictability under Qwen differs by −0.012 to −0.021 nats on common tokens;
  - wiki15, whose human half shares 99% of its shards with base, already shows −0.012, so most of that difference is sampling noise.
  - The synthetic halves differ about 10× more (−0.11 to −0.28).

  It is unlikely to explain α35's word-level losses, but it is fully collinear with α. The synthesis correlations involving `human_refnll_delta` and `human_dump_tvd` are therefore not separable from α and should not be read causally.
- **Doc length** (synthetic tokens/doc: 448 base, 450–468 at α15, 550–577 at α35) is also collinear with α. It is not a long-context effect, though: see section 2.1.
- **Single seed.** The cross-architecture replication in section 5 is the strongest evidence we have that the per-token patterns are data-driven.

## Bottom line

1. **Gains in all arms concentrate on rare and hard tokens and whitespace, with no cost on the easiest tokens and no long-context component.** That mechanism is common to register and wiki, at both α, in both architectures.
2. **α35 is not "more of the same".**
   - Its corpora are 2.5–3× less predictable.
   - It keeps or extends the hard-tail and layout gains.
   - But it **loses prediction of function words and of word tokens in prose domains** (FineWeb, FreeLaw, Gutenberg), which makes it worse overall.
3. **Wiki vs register:** the corpora look the same on frequency and predictability. Wiki's perplexity advantage is domain content (arXiv, Wikipedia), not a different mechanism. Register's benchmark advantage has to come from something these LM-loss measures don't capture, such as QA/table formatting.

## Next steps

- **Temperature-matched baseline:** an unsteered generation tuned to α15's reference-NLL shift. It is still the cleanest test of "direction vs just diversity", now with the added prediction that a matched α35-level temperature should also hurt function words.
- **Matched human half** for one α35 arm, to close the collinearity.
- **α between 15 and 35** (e.g. 25), to locate the peak of the inverted U.
