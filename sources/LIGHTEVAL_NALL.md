# Paper-style evaluation on the full evaluation sets (lighteval, 3-shot)

The setup is the same as in [`LIGHTEVAL_N1000.md`](LIGHTEVAL_N1000.md): finephrase task definitions, cloze formulation, 3-shot, `prob_norm_token`, and paired bootstrap tests. The difference is that this run uses **every question in each evaluation set** instead of the paper's 1,000-question cap. It was run on 2026-10-30 as SLURM job 210037, and each model took 38–54 minutes on one GH200. The qwen wiki models (`wiki_a15`, `wiki_a35`) were added on 2026-10-31 (job 210161).
The **qwen2 2.9B** models (`base`, `register_a35`, `wiki_a35`; same 20B-token mixes as the 1.7B models) were added on 2026-11-04 (jobs 216313 and 216343, about 65 minutes per model); see finding 6.

- **Per-family reports with tables for every benchmark vs the baseline:** [`llama_1b/lighteval_nall/REPORT.md`](llama_1b/lighteval_nall/REPORT.md), [`qwen2_1.7b/lighteval_nall/REPORT.md`](qwen2_1.7b/lighteval_nall/REPORT.md), [`qwen2_2.9b/lighteval_nall/REPORT.md`](qwen2_2.9b/lighteval_nall/REPORT.md).
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
   - **qwen 2.9B:** same sign in 27 of 30 cases (correlation 0.92). None of its 14 significant results was lost; 5 more became significant.

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
   | qwen2_2.9b | register_a35 | +0.47 | **+0.71** [+0.63, +0.79] | 1.4–4.4× |
   | | wiki_a35 | +0.28 | +0.26 [+0.18, +0.34] | 0.5–1.6× |

   The conclusion is the same as before. **qwen register_a35 is the only gain clearly above the paper's seed noise** (0.16–0.49). The other gains (all llama models, qwen register_a15 and both qwen wiki models) are within about one to four seed SDs, and llama wiki_a35 shows no gain at all.

   **Direct comparisons between interventions** (macro Δ, paired bootstrap; from `contrasts.csv`):

   | Contrast | llama_1b | qwen2_1.7b | qwen2_2.9b |
   |---|---:|---:|---:|
   | register − wiki, alpha 15 | +0.12 [+0.06, +0.19] | +0.29 [+0.22, +0.36] | (not trained) |
   | register − wiki, alpha 35 | +0.38 [+0.31, +0.44] | +0.38 [+0.31, +0.46] | **+0.45** [+0.38, +0.52] |
   | a35 − a15, register | −0.06 [−0.13, +0.01] | +0.37 [+0.30, +0.45] | (not trained) |
   | a35 − a15, wiki | −0.32 [−0.38, −0.26] | +0.28 [+0.21, +0.35] | (not trained) |

   - **Register data beats wiki data in all five comparisons** (three families, both alphas), by +0.12 to +0.45. The widest margin is at qwen 2.9B. This is the most consistent design-level result we have. Each margin is still within the seed noise of comparing two single-seed models (√2 × seed SD ≈ 0.23–0.69), but having all four point the same way makes chance a less likely explanation. **But the lead is task-specific:** on the 11 tasks shared with lm-eval, register and wiki are about level (qwen a35: +0.45 vs +0.43). Register's advantage comes from table QA (WikiTableQ, TREB-QA) and SQuADv2, where wiki is worse. On the lm-eval suite, the qwen wiki models are actually the more consistent improvers (see [`qwen2_1.7b/REPORT.md`](qwen2_1.7b/REPORT.md), finding 5).
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

6. **qwen2 2.9B (added 2026-11-04).**
   - **Macro:** register_a35 **+0.71** [+0.63, +0.79], about 1.4–4.4 paper seed SDs. That's below qwen 1.7B register_a35 (+1.03) but above every llama model. wiki_a35 is +0.26 [+0.18, +0.34].
   - **The full sets raised register_a35's gain from +0.47 to +0.71,** the largest change between the two runs of any model. Almost all of it is DROP: +2.80 on the first 1,000 questions, **+4.99 †** on all 9,536. That's the largest single gain in the data. The first 1,000 DROP questions underestimated the gain for every family.
   - **Register beats wiki by +0.45** [+0.38, +0.52], the widest margin of the three families. It comes from reading comprehension (+1.84 vs +0.34; DROP +4.99 vs +2.23) and tables (+0.92 vs −0.29).
   - **The robust effects repeat:**
     - DROP gains in both models;
     - XCSQA +1.41 † / +1.06 †;
     - GSM8K +1.25 † / +0.96 †. GSM8K improves at both qwen sizes, so the GSM8K loss is llama-specific;
     - the register_a35 LAMBADA cost (−1.06 †), now in all three families (llama −0.86 †, qwen 1.7B −0.86 †);
     - SQuADv2 losses (−1.31 † / −1.55 †).
   - **What differs from 1.7B:**
     - Table gains are smaller: +0.92 / −0.29, against +2.98 / +2.52. TREB-QA doesn't move.
     - Language understanding is slightly negative: WinoGrande −0.43 † / −0.40 †, HellaSwag −0.05 † for register.
     - TriviaQA improves for both models (+0.43 † / +0.52 †).
   - **Probability vs accuracy:** they agree on 2 of 8 multiple-choice tasks for register_a35 and 7 of 8 for wiki_a35. Register_a35's probability changes on those tasks are mostly near zero, while its accuracy rises on ARC-Easy, XCSQA and HellaSwag; finding 7 explains what that means.
   - **Held-out perplexity** (from `final_eval.json`; the evaluation bug is fixed in these runs). Mean gain over 7 domains, in centi-nats: register_a35 +0.96, wiki_a35 +1.43. That compares with +1.07 / +1.66 at qwen 1.7B and +1.33 / +1.96 at llama. The benefit shrinks gradually with size, and FineWeb is the one domain worse than base (−0.77 for both).
   - **Caveat:** one seed per model. The 2.9B models saw the same 20B tokens as the smaller ones, about 7.5 tokens per parameter against about 12 at 1.7B, so they are the least trained.

7. **What the models learn, seen through accuracy (multiple-choice tasks).**
   Only the 8 multiple-choice tasks have an accuracy (`acc_norm_token`): the free-answer tasks (DROP, GSM8K, WikiTableQ, TriviaQA, SQuADv2, LAMBADA, TREB-QA) are scored by probability only. On those 8 tasks the two metrics answer different questions:
   - **probability** (`prob_norm_token`): how much probability the model puts on the correct answer;
   - **accuracy**: whether the correct answer beats the wrong options.

   Reading them together separates two kinds of change:
   - **Probability up, accuracy flat:** the wrong options gained too. This is a general boost to answer-like text, a format effect.
   - **Accuracy up, probability flat:** the model ranks the correct option higher more often, without being more confident. This is better discrimination.

   Each cell is accuracy Δ / probability Δ in points; † = significant after BH correction. Accuracy is binary and most of these sets are small (500–2,400 questions, HellaSwag 10,042), so it takes about 2–3 points to reach significance. A non-significant accuracy cell means "too small to see", not zero.

| task (base acc llama / q1.7B / q2.9B) | llama reg_a15 | llama reg_a35 | llama wiki_a15 | llama wiki_a35 | q1.7B reg_a15 | q1.7B reg_a35 | q1.7B wiki_a15 | q1.7B wiki_a35 | q2.9B reg_a35 | q2.9B wiki_a35 |
|---|---|---|---|---|---|---|---|---|---|---|
| XCSQA (38.9 / 44.6 / 46.6) | +2.90† / +0.88† | +5.90† / +1.21† | +4.00† / +0.70† | +4.50† / +1.28† | +0.10 / +0.10 | +2.80 / +0.60† | +2.70 / +0.56† | +1.40 / +0.34† | +4.40† / +1.41† | +0.80 / +1.06† |
| HellaSwag (43.4 / 46.3 / 49.7) | +1.01† / +0.15† | +0.87† / +0.07† | +0.58 / +0.04† | +0.69 / +0.09† | +0.95† / +0.18† | +1.49† / +0.15† | +0.79 / +0.23† | +1.46† / +0.21† | +1.00† / -0.05† | +1.47† / +0.06† |
| ARC-Easy (52.3 / 56.6 / 57.6) | +1.01 / +0.56† | +1.60 / +0.84† | +2.40† / +0.54† | +2.53† / +1.04† | +1.35 / +0.33† | +0.76 / +0.42† | +1.14 / +0.48† | +1.35 / +0.41† | +3.11† / -0.07 | +3.11† / +0.45† |
| WinoGrande (53.1 / 52.4 / 53.9) | -0.55 / +0.39† | +0.47 / +0.28† | -0.55 / +0.57† | -0.87 / +0.42† | +1.03 / -0.04 | +0.95 / +0.34† | -0.24 / +0.04 | +1.26 / +0.28† | -1.03 / -0.43† | -1.03 / -0.40† |
| PIQA (70.7 / 72.1 / 73.2) | +0.76 / +0.23† | +0.92 / +0.08† | +1.90† / +0.25† | +0.98 / +0.34† | +1.09 / +0.09† | +0.27 / +0.00 | +0.44 / +0.12† | +0.65 / +0.19† | +0.82 / -0.00 | +0.71 / +0.14† |
| ARC-Challenge (25.9 / 27.1 / 29.7) | +0.17 / +0.40† | +1.11 / +0.33† | -1.02 / +0.31† | +0.34 / +0.35† | -0.60 / +0.23† | +1.71 / +0.37† | +0.51 / +0.30† | +1.37 / +0.38† | -0.43 / +0.11 | +0.68 / +0.06 |
| OpenBookQA (26.0 / 27.8 / 28.4) | +0.20 / +0.18† | +0.80 / +0.28† | +0.20 / +0.12† | -0.20 / +0.20† | -1.00 / +0.20† | +2.60 / +0.53† | -1.00 / +0.38† | +1.00 / +0.42† | +0.00 / +0.06 | -0.40 / +0.14 |
| MMLU-Redux (30.5 / 31.1 / 32.4) | +0.28 / +0.28† | +0.37 / +0.33† | +0.19 / +0.06 | +0.76 / +0.35† | +0.12 / +0.14† | -0.07 / +0.28† | -0.28 / +0.13 | +0.30 / -0.00 | -0.98 / +0.10 | +0.04 / +0.08 |

   How many of the 8 tasks change significantly:

| | llama reg_a15 | llama reg_a35 | llama wiki_a15 | llama wiki_a35 | q1.7B reg_a15 | q1.7B reg_a35 | q1.7B wiki_a15 | q1.7B wiki_a35 | q2.9B reg_a35 | q2.9B wiki_a35 |
|---|---|---|---|---|---|---|---|---|---|---|
| probability: significant gains / losses (of 8) | 8 / 0 | 8 / 0 | 7 / 0 | 8 / 0 | 6 / 0 | 7 / 0 | 6 / 0 | 7 / 0 | 1 / 2 | 4 / 1 |
| accuracy: significant gains / losses (of 8) | 2 / 0 | 2 / 0 | 3 / 0 | 2 / 0 | 1 / 0 | 1 / 0 | 0 / 0 | 1 / 0 | 3 / 0 | 2 / 0 |

   - **llama 1B and qwen 1.7B: mostly a format effect.** Probability rises on 6–8 of the 8 tasks in every model, but accuracy changes significantly on only 0–3. The steered data makes answer-like text in general more likely, wrong options included, and rarely changes which option wins. The exceptions are real gains in choosing correctly:
     - **XCSQA** for all four llama models (+2.9 to +5.9);
     - **HellaSwag** in 5 of 8 models;
     - **ARC-Easy** for both llama wiki models, **PIQA** for llama wiki_a15.
   - **qwen 2.9B: mostly better choices.** The general probability boost is gone: register_a35 has 1 significant gain and 2 losses out of 8. Yet it has the most significant accuracy gains of any model:
     - **ARC-Easy +3.11 †** at a probability change of −0.07;
     - **XCSQA +4.40 †**;
     - **HellaSwag +1.00 †** while probability falls (−0.05 †).

     The larger model doesn't put more probability on answers; it picks the right one more often. wiki_a35 shows the same pattern on ARC-Easy (+3.11 †) and HellaSwag (+1.47 †).
   - **XCSQA is the most robust real gain.** Both metrics rise in most models, and accuracy gains are large (+2.8 to +5.9 for register_a35 in all three families). Choosing the right commonsense concept really improves; this is not only formatting.
   - **The prose cost is mostly lost confidence, not worse judgement.**
     - At 2.9B, HellaSwag probability falls while accuracy rises. The model gives ordinary continuations a little less probability but still ranks the plausible one higher more often.
     - That is the multiple-choice counterpart of the LAMBADA losses and the worse held-out FineWeb perplexity (finding 6). Plain-prose likelihood drops; commonsense choice doesn't.
     - **WinoGrande at 2.9B** is the one exception: both metrics point down (−1.03 / −0.43 † for both models). Accuracy isn't significant, but it is the only candidate for a real reasoning cost.
   - **Knowledge is unchanged.** Accuracy on MMLU-Redux, OpenBookQA, ARC-Challenge and PIQA is not significant in any model except llama wiki_a15 PIQA. The steered rewrites change how content is phrased, not which facts it contains.
   - **Register vs wiki at 2.9B:** identical on ARC-Easy accuracy (+3.11) and close on HellaSwag. Register's lead comes from XCSQA (+4.40 vs +0.80) and the free-answer tasks (DROP, tables).
   - **Overall picture, combined with the free-answer tasks:**
     - **Better:** turning a question into a short, specific answer. That covers DROP numbers and spans, table cells, GSM8K answers, XCSQA concepts and TriviaQA entities. At the smaller sizes this shows up as higher probability on answer-like text in general; at 2.9B as better choices.
     - **Worse:** likelihood of plain running prose (LAMBADA, FineWeb perplexity, HellaSwag probability at 2.9B, possibly WinoGrande) and copying exact spans (SQuADv2).
     - **Unchanged:** knowledge.
   - **Caveat:** one training seed per model. The size pattern (format effect at 1–1.7B, better choices at 2.9B) rests on one model per size, so it is a hypothesis to check with the 2.9B seed, not a finding.
   - Regenerate the table from `<family>/lighteval_nall/task_deltas.csv`, which has rows for both metrics.

### Bottom line
- **With 1,000 questions per task you reach the same conclusions as with the full sets.** That makes the 1k cap fine for quick screening, just as the paper uses it. For final numbers, use the full sets: DROP and WikiTableQ in particular move by 0.5–1.3 points.
- **Robust effects, all present in both runs:**
  - **DROP** gains in 9 of 10 intervention models (not qwen 1.7B wiki_a15), up to +4.99 for qwen 2.9B register_a35;
  - **table QA** gains in 9 of 10 (large on qwen 1.7B, smaller at 2.9B, none for qwen 2.9B wiki_a35);
  - **register data beats wiki data** in all five family × alpha comparisons, with wiki also costing SQuADv2;
  - **XCSQA** gains in all 10;
  - **llama GSM8K** losses (worst at alpha 35); GSM8K improves at both qwen sizes;
  - **LAMBADA** losses for register_a35 in all three families;
  - small gains on ARC, MMLU-Redux, OpenBookQA and WinoGrande for the smaller models. At 2.9B these are flat, and WinoGrande turns negative.
- **What the models learn (finding 7):**
  - **Better:** answering questions with a short, specific answer. At 1–1.7B this shows up mostly as higher probability on answer-like text, with real accuracy gains on XCSQA and HellaSwag. At 2.9B it shows up as more correct choices (ARC-Easy, XCSQA, HellaSwag) without higher probability.
  - **Worse:** likelihood of plain prose, and copying exact spans.
  - **Unchanged:** knowledge.
- **The deciding question is still training-seed noise.** Measuring our own seed SD (2 extra seeds for qwen base and qwen register_a35, ideally also llama base and register_a15, plus one wiki model to test register vs wiki; at 2.9B, one extra seed of base and register_a35 would show whether the smaller gain there is real) is what would turn the macro gains into claims. Evaluating more questions won't change that.

## Figures

**llama_1b:** macro aggregate, per-task change, and the 1k run vs the full sets

![llama macro](llama_1b/lighteval_nall/figures/aggregate_macro.png)
![llama forest](llama_1b/lighteval_nall/figures/forest_prob_norm_token.png)
![llama compare](llama_1b/lighteval_nall/figures/compare_n1000.png)

**qwen2_1.7b:** macro aggregate, per-task change, and the 1k run vs the full sets

![qwen macro](qwen2_1.7b/lighteval_nall/figures/aggregate_macro.png)
![qwen forest](qwen2_1.7b/lighteval_nall/figures/forest_prob_norm_token.png)
![qwen compare](qwen2_1.7b/lighteval_nall/figures/compare_n1000.png)

**qwen2_2.9b:** macro aggregate, per-task change, and the 1k run vs the full sets

![qwen 2.9b macro](qwen2_2.9b/lighteval_nall/figures/aggregate_macro.png)
![qwen 2.9b forest](qwen2_2.9b/lighteval_nall/figures/forest_prob_norm_token.png)
![qwen 2.9b compare](qwen2_2.9b/lighteval_nall/figures/compare_n1000.png)

## Caveats
- **One training seed per configuration.** The seed SD comes from the paper (a different tokenizer, data and model), so it is only a rough reference.
- **The macro score is not the paper's exact number,** because the full sets weight WikiTableQ, TriviaQA and the others differently from the 1k cap. Within this run, all comparisons are like for like.
