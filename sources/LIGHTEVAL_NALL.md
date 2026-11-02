# Paper-style evaluation on the full evaluation sets (lighteval, 3-shot)

The setup is the same as in [`LIGHTEVAL_N1000.md`](LIGHTEVAL_N1000.md): finephrase task definitions, cloze formulation, 3-shot, `prob_norm_token`, and paired bootstrap tests. The difference is that this run uses **every question in each evaluation set** instead of the paper's 1,000-question cap. It was run on 2026-10-30 as SLURM job 210037, and each model took 38–54 minutes on one GH200. The qwen wiki models (`wiki_a15`, `wiki_a35`) were added on 2026-10-31 (job 210161).

- **Per-family reports with tables for every benchmark vs the baseline:** [`llama_1b/lighteval_nall/REPORT.md`](llama_1b/lighteval_nall/REPORT.md), [`qwen2_1.7b/lighteval_nall/REPORT.md`](qwen2_1.7b/lighteval_nall/REPORT.md).
- **Comparison with the 1k run:** `comparison_vs_n1000.csv`, `aggregates_vs_n1000.csv` and `figures/compare_n1000.png` in each `lighteval_nall/` folder.
- **Regenerate:**
  - `bash container_python.sh analyze_lighteval.py <family> nall`
  - `bash container_python.sh compare_lighteval_tags.py <family> n1000 nall`
  - `python3 report_lighteval.py <family> nall`

**Question counts** (the 1k run capped these at 1,000):

| Task | Questions |
|---|---|
| WikiTableQ | 18,486 |
| TriviaQA | 17,944 |
| HellaSwag | 10,042 |
| DROP | 9,536 |
| SQuADv2 | 5,928 |
| LAMBADA | 5,153 |
| ARC-Easy | 2,376 |
| PIQA | 1,838 |
| GSM8K | 1,319 |
| WinoGrande | 1,267 |
| ARC-Challenge | 1,172 |

The rest were already complete in the 1k run, so their numbers are identical in both: XCSQA (1,000), OpenBookQA (500), TREB-QA (490) and MMLU-Redux (5,431 over 57 subjects).

---

## Key findings

1. **The full sets confirm the 1k run.**
   - Per-task changes vs the baseline have the same sign in **58 of 60** cases for llama (correlation 0.94) and **55 of 60** for qwen (0.98).
   - **No result that was significant at 1k lost significance or flipped sign.** The sign flips that did happen were all non-significant changes near zero.
   - 18 more task changes became significant (7 llama, 11 qwen), and the CIs are roughly 30–40% narrower.

2. **The macro scores barely changed:**

   | Family | Model | Macro Δ, 1k run | Macro Δ, full sets [95% CI] | Δ ÷ paper seed SD |
   |---|---|---:|---:|---:|
   | llama_1b | register_a15 | +0.33 | **+0.45** [+0.38, +0.51] | 0.9–2.8× |
   | | register_a35 | +0.38 | +0.38 [+0.31, +0.46] | 0.8–2.4× |
   | | wiki_a15 | +0.37 | +0.33 [+0.26, +0.40] | 0.7–2.0× |
   | | wiki_a35 | +0.06 | +0.01 [−0.06, +0.08] | ≈ 0 |
   | qwen2_1.7b | register_a15 | +0.58 | **+0.65** [+0.58, +0.72] | 1.3–4.1× |
   | | register_a35 | +1.02 | **+1.03** [+0.95, +1.11] | 2.1–6.4× |
   | | wiki_a15 | +0.33 | +0.36 [+0.29, +0.43] | 0.7–2.3× |
   | | wiki_a35 | +0.57 | +0.64 [+0.57, +0.72] | 1.3–4.0× |

   The conclusion is the same as before. **qwen register_a35 is the only gain clearly above the paper's seed noise** (0.16–0.49). The other gains (all llama models, qwen register_a15 and both qwen wiki models) are within about one to four seed SDs, and llama wiki_a35 shows no gain at all.

   **Direct comparisons between interventions** (macro Δ, paired bootstrap; from `contrasts.csv`):

   | Contrast | llama_1b | qwen2_1.7b |
   |---|---:|---:|
   | register − wiki, alpha 15 | +0.12 [+0.06, +0.19] | +0.29 [+0.22, +0.36] |
   | register − wiki, alpha 35 | +0.38 [+0.31, +0.44] | +0.38 [+0.31, +0.46] |
   | a35 − a15, register | −0.06 [−0.13, +0.01] | +0.37 [+0.30, +0.45] |
   | a35 − a15, wiki | −0.32 [−0.38, −0.26] | +0.28 [+0.21, +0.35] |

   - **Register data beats wiki data in all four comparisons** (both architectures, both alphas), by +0.12 to +0.38. This is the most consistent design-level result we have. Each margin is still within the seed noise of comparing two single-seed models (√2 × seed SD ≈ 0.23–0.69), but having all four point the same way makes chance a less likely explanation. **But the lead is task-specific:** on the 11 tasks shared with lm-eval, register and wiki are about level (qwen a35: +0.45 vs +0.43). Register's advantage comes from table QA (WikiTableQ, TREB-QA) and SQuADv2, where wiki is worse. On the lm-eval suite, the qwen wiki models are actually the more consistent improvers (see [`qwen2_1.7b/REPORT.md`](qwen2_1.7b/REPORT.md), finding 5).
   - **Alpha 35 helps on qwen** (both data sources, +0.28 / +0.37) **but not on llama** (−0.06 / −0.32).

3. **What changed with more questions:**
   - **DROP gains are larger** with all 9,536 questions. On llama, register_a15 goes from +0.86 to +2.13 and wiki_a15 from +1.05 to +1.51; on qwen, register_a15 goes from +1.33 to +2.18. The first 1,000 DROP questions underestimated the effect. DROP is now **+1.5 to +2.8 in 7 of 8 intervention models**, the most consistent gain in the data. The exception is **qwen wiki_a15 (−0.71 †)**.
   - **WikiTableQ gains are smaller** with all 18,486 questions. For example, llama wiki_a15 drops from +1.35 to +0.39 and register_a35 from +3.05 to +2.52. On qwen they shrink only slightly (+4.69 → +4.32 for register_a35). Table QA is still the largest gain on qwen.
   - **TriviaQA is now positive and significant for all four llama models** (+0.17 to +0.79), and for qwen register_a15 (+0.27) and both qwen wiki models (+0.15 / +0.22).
   - **LAMBADA: register_a35 is significantly worse in both architectures** (llama −0.86 †, qwen −0.86 †). In the 1k run this showed only on qwen. Together with the worse LAMBADA perplexity in lm-eval (llama +7.6%, qwen +5.3%), this is now a **consistent cost of alpha 35 with register data, in both architectures**. The qwen wiki models don't show it (wiki_a15 +0.53 †, wiki_a35 −0.08).
   - **SQuADv2:** wiki data hurts here. llama wiki_a35 (−0.94 †) and both qwen wiki models (−0.96 / −1.57 †) are significantly worse, while the register models and llama wiki_a15 gain +0.5 to +1.5.

4. **The llama GSM8K loss is stable:** −0.55 (register_a15), −1.11 (wiki_a15), −2.35 (register_a35), −2.51 (wiki_a35). These are among the largest effects in the data, and alpha 35 costs the most. Qwen register still gains (+0.47 / +0.87); qwen wiki gains at alpha 15 (+0.80) and is unchanged at alpha 35 (−0.03).

5. **Accuracy now agrees better.** Probability and accuracy move the same way on 6–8 of 8 tasks for llama and 5–7 of 8 for qwen.
   - Accuracy changes that survive multiple-test correction:
     - **XCSQA:** all four llama models, +2.9 to +5.9.
     - **ARC-Easy:** llama wiki_a15 and wiki_a35, +2.4 / +2.5.
     - **PIQA:** llama wiki_a15, +1.9.
     - **HellaSwag:** both register models in both architectures, plus qwen wiki_a35, +0.9 to +1.5.
   - So with enough questions, the probability gains start to show up as more correct answers.

### Bottom line
- **With 1,000 questions per task you reach the same conclusions as with the full sets.** That makes the 1k cap fine for quick screening, just as the paper uses it. For final numbers, use the full sets: DROP and WikiTableQ in particular move by 0.5–1.3 points.
- **Robust effects, all present in both runs:**
  - **DROP** gains in 7 of 8 intervention models (not qwen wiki_a15);
  - **table QA** gains in all 8 (large on qwen);
  - **register data beats wiki data** in all four architecture × alpha comparisons, with wiki also costing SQuADv2;
  - **llama GSM8K** losses (worst at alpha 35);
  - **LAMBADA** losses for register_a35 in both architectures;
  - small but consistent gains on ARC, MMLU-Redux, OpenBookQA, WinoGrande and XCSQA.
- **The deciding question is still training-seed noise.** Measuring our own seed SD (2 extra seeds for qwen base and qwen register_a35, ideally also llama base and register_a15, plus one wiki model to test register vs wiki) is what would turn the macro gains into claims. Evaluating more questions won't change that.

## Figures

**llama_1b:** macro aggregate, per-task change, and the 1k run vs the full sets

![llama macro](llama_1b/lighteval_nall/figures/aggregate_macro.png)
![llama forest](llama_1b/lighteval_nall/figures/forest_prob_norm_token.png)
![llama compare](llama_1b/lighteval_nall/figures/compare_n1000.png)

**qwen2_1.7b:** macro aggregate, per-task change, and the 1k run vs the full sets

![qwen macro](qwen2_1.7b/lighteval_nall/figures/aggregate_macro.png)
![qwen forest](qwen2_1.7b/lighteval_nall/figures/forest_prob_norm_token.png)
![qwen compare](qwen2_1.7b/lighteval_nall/figures/compare_n1000.png)

## Caveats
- **One training seed per configuration.** The seed SD comes from the paper (a different tokenizer, data and model), so it is only a rough reference.
- **The macro score is not the paper's exact number,** because the full sets weight WikiTableQ, TriviaQA and the others differently from the 1k cap. Within this run, all comparisons are like for like.
