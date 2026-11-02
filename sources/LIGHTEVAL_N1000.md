# Paper-style evaluation (lighteval, 3-shot, 1,000-question cap)

The same setup as finephrase (arXiv 2604.13977):
- lighteval 0.9.1.dev0, using the finephrase task definitions verbatim;
- cloze formulation (the model scores the full answer text, not a letter), 3-shot;
- the paper's metric `prob_norm_token`: the token-normalised probability of the correct answer;
- `--max-samples 1000` per task, as in their runs.

This covers all 12 paper tasks plus ARC-Challenge, TriviaQA and LAMBADA (0-shot). It was run on 2026-10-30 as SLURM job 210016, and each model took 17–24 minutes on one GH200. The qwen wiki models (`wiki_a15`, `wiki_a35`) were added on 2026-10-31 (job 210160).

- **Per-family reports with tables for every benchmark vs the baseline:** [`llama_1b/lighteval_n1000/REPORT.md`](llama_1b/lighteval_n1000/REPORT.md), [`qwen2_1.7b/lighteval_n1000/REPORT.md`](qwen2_1.7b/lighteval_n1000/REPORT.md) (built by `python3 report_lighteval.py <family> n1000`).
- Per-family outputs: [`llama_1b/lighteval_n1000/`](llama_1b/lighteval_n1000), [`qwen2_1.7b/lighteval_n1000/`](qwen2_1.7b/lighteval_n1000). Each has `task_deltas.csv`, `aggregates.csv`, `summary.json` and `figures/`.
- Produced by `bash container_python.sh analyze_lighteval.py <family> n1000`.

**Statistics.** Every model answers the same 1,000 questions, so the comparisons against the baseline are paired:
- 95% CIs and p-values come from a paired bootstrap (2,000 resamples, questions drawn jointly for all models), and the same resampling carries through to the category and macro aggregates.
- `†` means Benjamini–Hochberg q < 0.05.
- The **macro score** is built exactly as in the paper: the mean of six category means. The categories are GK = ARC-E, MMLU-Redux; RC = SQuADv2, DROP; RES = OBQA, PIQA, XCSQA; NLU = HellaSwag, WinoGrande; MATH = GSM8K; TABLE = WikiTableQ, TREB-QA.
- Scores are shown ×100.
- The task means my script computes match lighteval's own results files exactly (checked automatically).

---

## Key findings

1. **The paired tests measure the benchmarks precisely, but that's only half the noise.** With 1,000 paired questions, gains of about 0.1–0.2 points are detectable, so almost every task delta is "significant". That is *evaluation* noise only. The relevant yardstick for a single training run is **seed-to-seed noise, which the paper measured at 0.16–0.49 points on this exact macro score** (their Table 21; 0.27 at 1.7B).

2. **Paper-style macro score vs the baseline:**

   | Family | Model | Macro | Δ [95% CI] | Δ ÷ paper seed SD |
   |---|---|---:|---:|---:|
   | llama_1b | base | 16.49 | | |
   | | register_a15 | 16.82 | +0.33 [+0.22, +0.44] | 0.7–2.1× |
   | | register_a35 | 16.88 | +0.38 [+0.27, +0.50] | 0.8–2.4× |
   | | wiki_a15 | 16.86 | +0.37 [+0.26, +0.48] | 0.8–2.3× |
   | | wiki_a35 | 16.56 | +0.06 [−0.05, +0.18] | 0.1–0.4× |
   | qwen2_1.7b | base | 17.62 | | |
   | | register_a15 | 18.20 | +0.58 [+0.46, +0.70] | 1.2–3.6× |
   | | **register_a35** | **18.63** | **+1.02 [+0.89, +1.14]** | **2.1–6.3×** |
   | | wiki_a15 | 17.95 | +0.33 [+0.22, +0.45] | 0.7–2.1× |
   | | wiki_a35 | 18.19 | +0.57 [+0.45, +0.70] | 1.2–3.6× |

   - **On llama, every gain is within about one to two seed SDs.** With one seed per configuration, you can't claim from this that any llama intervention beats the baseline.
   - **qwen register_a35 (+1.02) is the only gain clearly larger than the paper's seed noise.** It is the best candidate to confirm with extra seeds. The two qwen wiki models (+0.33, +0.57) are in the same range as the llama models.
   - For scale, the paper's headline effect was +4.7 to +5.2, which is 14–32 seed SDs. Our effects are 5–15× smaller.

   **Direct comparisons between interventions** (macro Δ, paired bootstrap; from `contrasts.csv`):

   | Contrast | llama_1b | qwen2_1.7b |
   |---|---:|---:|
   | register − wiki, alpha 15 | −0.04 [−0.14, +0.06] | +0.25 [+0.13, +0.37] |
   | register − wiki, alpha 35 | +0.32 [+0.22, +0.42] | +0.44 [+0.32, +0.56] |
   | a35 − a15, register | +0.05 [−0.05, +0.16] | +0.43 [+0.31, +0.56] |
   | a35 − a15, wiki | −0.31 [−0.41, −0.20] | +0.24 [+0.13, +0.37] |

   - **Register data beats wiki data** in 3 of 4 comparisons; llama at alpha 15 is a tie. Every margin is about 0.25–0.45, which is no larger than the seed noise of comparing two single-seed models (√2 × seed SD ≈ 0.23–0.69). **But the lead is task-specific:** on the 11 tasks shared with lm-eval, register and wiki are about level (qwen a35: +0.45 vs +0.43). Register's advantage comes from table QA (WikiTableQ, TREB-QA) and SQuADv2, where wiki is worse. On the lm-eval suite, the qwen wiki models are actually the more consistent improvers (see [`qwen2_1.7b/REPORT.md`](qwen2_1.7b/REPORT.md), finding 5).
   - **Alpha 35 helps on qwen** (for both data sources) **but not on llama**.

3. **Where the gains come from:**
   - **Table tasks:** WikiTableQ improves in all 8 intervention models: +0.9 to +3.1 on llama and +2.3 to +4.7 on qwen. TREB-QA improves by up to +1.7.
   - **Reading comprehension:** DROP improves in 7 of 8 models (+0.9 to +2.6, largest at alpha 35). The exception is **qwen wiki_a15**, which is worse on DROP (−0.84 †). Both qwen wiki models are also worse on SQuADv2 (−1.3 / −1.7 †), which pulls qwen wiki's reading-comprehension score below register's.
   - **XCSQA:** +0.7 to +1.3 on llama, +0.1 to +0.6 on qwen.
   - **Commonsense tasks:** HellaSwag, PIQA and WinoGrande move by only about 0.0–0.5.
   - So the paraphrased data mostly helps tasks with structured or extractive answers, which is the same pattern the paper reports for its table and QA formats.

4. **GSM8K depends on the architecture.** Here the GSM8K metric is the probability of the final numeric answer.
   - On llama, **all four interventions are significantly worse**: −0.7 (register_a15), −1.2 (wiki_a15), −2.4 (register_a35), −2.6 (wiki_a35). Alpha 35 costs the most.
   - On qwen, register is *better* (+0.5 / +0.9) and wiki is better at alpha 15 (+0.75) and unchanged at alpha 35 (−0.04).
   - This is the largest single loss in the data, and it's worth checking whether the llama paraphrase data drops or rewrites numbers.

5. **Alpha 35 vs alpha 15:**
   - On qwen, a35 beats a15 for both data sources (macro: register +1.02 vs +0.58, wiki +0.57 vs +0.33).
   - On llama, a35 wins on tables, DROP, ARC and XCSQA but loses on GSM8K and the language-understanding tasks (HellaSwag, WinoGrande). Net, the macro score is a wash for register and worse for wiki.

6. **How this compares with the earlier lm-eval reports:**
   - **DROP gains are confirmed** in both architectures, except qwen wiki_a15.
   - **The SQuADv2 jump for qwen register_a15 in lm-eval** (+7.4 F1 on answerable questions) **is not reproduced** here: +0.44, not significant. That gain was probably about how answers were generated and formatted, not about knowledge.
   - **TriviaQA:** small gains on llama (a35 and both wiki models, +0.5 to +0.7), none on qwen (−0.3 to 0.0). In lm-eval, qwen register_a35 had +0.98 exact match at 0-shot.
   - **LAMBADA:** qwen register_a35 is significantly worse (−1.22 †), which matches its worse LAMBADA perplexity in lm-eval (+5.3%). The qwen wiki models are not significantly different (−0.01 / −0.45). On llama, register_a35 is unchanged (0.0), so the perplexity tendency seen in lm-eval is not confirmed there.
   - **MMLU is now off chance level** (MMLU-Redux scored on the full answer text: acc_norm about 30–31%), and every intervention has a small positive probability gain (+0.06 to +0.35).

7. **Accuracy agrees only partly.** Token-normalised accuracy has the same sign as `prob_norm_token` on 4–7 of the 8 multiple-choice tasks per model (5 / 7 for the qwen wiki models).
   - Most accuracy deltas are not significant at n = 1,000. The exception is XCSQA (+2.9 to +5.9 on llama, +2.8 for qwen a35), plus ARC-Easy for llama wiki_a35 (+3.9) and HellaSwag on qwen (+1.9 / +2.2).
   - This is the expected pattern: at this scale, probability mass moves before the top answer flips. It's also why accuracy alone looked so noisy in the lm-eval reports.

### What this means
- **The single-seed problem is now the bottleneck, not evaluation noise.** Only the qwen register_a35 macro gain, plus several per-task gains (tables, DROP, llama GSM8K losses), are large compared with the paper's seed SD.
- **Next steps, in order:**
  1. The full-set run (`MAX_SAMPLES` unset), to see whether the per-task picture holds with more questions.
  2. 2 extra seeds for **qwen base and qwen register_a35** (and ideally llama base and register_a15), to measure *our own* seed SD instead of borrowing the paper's.

## Figures

**llama_1b:** macro aggregate, and per-task change with paired 95% CIs

![llama macro](llama_1b/lighteval_n1000/figures/aggregate_macro.png)
![llama forest](llama_1b/lighteval_n1000/figures/forest_prob_norm_token.png)

**qwen2_1.7b:** macro aggregate, per-task change, and probability vs accuracy

![qwen macro](qwen2_1.7b/lighteval_n1000/figures/aggregate_macro.png)
![qwen forest](qwen2_1.7b/lighteval_n1000/figures/forest_prob_norm_token.png)
![qwen prob vs acc](qwen2_1.7b/lighteval_n1000/figures/prob_vs_acc.png)

## Caveats
- **One training seed per configuration.** The seed SD used above is the paper's (a 1.2B Qwen2 architecture with a different tokenizer and data), so it is only a rough reference for our models.
- **The 1,000-question cap** follows the paper. OpenBookQA (500) and TREB-QA (490) have fewer questions; MMLU-Redux has 5,431 questions over 57 subjects.
- **The paper's absolute scores aren't comparable to ours** (different tokenizer, data and training budget). Only the setup is the same.
