# Benchmark-text attribution and corpus format: findings

This tests the two predictions of the working hypothesis "benchmark gains come from format/genre exposure: register steering makes the FAQ paraphrases look more like the benchmarks". Both were run on 2026-10-31.

| Step | Script | Output | Job |
|---|---|---|---|
| 6: all 10 models on the exact lighteval prompts + gold answers | `scripts/06_benchmark_losses.sbatch` | `outputs/benchmark/SUMMARY.md` and CSVs | 210319 (12 min) |
| 7: format profile of the 5 synthetic corpora (+ held-out FineWeb) | `scripts/07_corpus_format.sbatch` | `outputs/format/FORMAT.md`, `format_docs.csv` | 210318 (1 min) |

**Step 6 setup.**
- **Items:** 1,500 each of WikiTableQ, SQuADv2 and DROP; all 490 TREB-QA; 2,000 each of LAMBADA and TriviaQA. These are sampled from the full-set (`nall`) lighteval details.
- **Scoring:** every model sees the identical 3-shot prompt plus the gold continuation. Every token is labelled table markup, table cell, passage, question, `Question:`/`Answer:` scaffold, whitespace, demo answer or gold answer.
- **Sanity check:** our gold-answer log-prob reproduces lighteval's per item with correlation 1.000 and a mean difference of 0.013 nats, for every model and task.
- **The subsample reproduces the published Δs.** Δ prob_norm_token on these items matches the full-set report, e.g. qwen reg35 WikiTableQ +4.6 vs +4.3 in `LIGHTEVAL_NALL.md`.

**Units.** Δ is in centi-nats (nats × 100); positive means better than base, or better for the first arm in a contrast.

## 1. Benchmark gains are gains on the answer, not on reading the input

On every task, the gain on the gold answer is 5–25 centi-nats. The gain on the evaluated example's own text (table, passage, question) is about +0.5 to +2.5. That is the same size as the held-out perplexity gains.

| (centi-nats) | llama reg35 | qwen reg35 | qwen wiki35 |
|---|---:|---:|---:|
| WikiTableQ answer / context | +14.9 / −0.2 | +24.9 / +0.6 | +21.9 / +1.1 |
| DROP answer / context | +18.3 / +0.1 | +19.6 / +0.6 | +14.2 / +1.5 |

Better modelling of the input does not predict better answers either. Across items, the regression of answer Δ on context Δ has:
- Spearman ρ of about 0 (|ρ| < 0.1 everywhere);
- an intercept that equals the full answer gain;
- a share of the answer gain "explained" by context of about 0 in nearly all cells.

**So the benchmark gains are not "the model understands tables or passages better".** They are "the model puts more probability on the short gold answer after `Answer:`".

## 2. Register beats wiki on answers while modelling the input *worse*

This is the clearest result. Register − wiki contrasts, answer vs the evaluated example's text:

| Task | llama @a15 | llama @a35 | qwen @a15 | qwen @a35 |
|---|---|---|---|---|
| WikiTableQ | −0.9 / −0.4† | **+8.3† / −1.4†** | **+2.6† / −0.7†** | **+3.0† / −0.6†** |
| DROP | −6.0† / +0.2† | **+4.4† / −2.3†** | **+7.6†** / +0.1 | **+5.4† / −0.9†** |
| SQuADv2 | +1.2 / −1.7† | **+6.2† / −2.3†** | +1.7 / −0.1 | +1.7 / −1.9† |
| TriviaQA | +0.4 / +0.6 | **+4.1† / −2.2†** | +2.0 / −2.6† | **+3.9† / −2.8†** |
| TREB-QA | −6.3† / −0.1 | **+7.7† / −0.6†** | −1.1 / −0.6† | −1.6 / −0.1 |
| LAMBADA | +0.5 / −0.3† | −8.6† / +0.1 | −3.7† / −0.5† | −4.7† / −0.3 |

Cells show answer / context; † marks a 95% CI that excludes 0.

- **Context:** register is almost always worse than wiki. It models table markup, Wikipedia passages and questions worse, which matches the held-out perplexity result (wiki is better on Wikipedia and arXiv).
- **Answers:** at α35, register is significantly better on 5 of 6 tasks in llama, and on 3 of 6 in qwen (4 of 6 positive).
- **Within register, the dissociation repeats with α:**
  - a35 − a15 *raises* answer prediction a lot: WikiTableQ +11.4 / +9.6 and DROP +13.2 / +11.8 (llama / qwen).
  - It *lowers* context prediction: −0.2 to −1.3, all †.

**The earlier assumption that "register's lead must be something loss doesn't measure" was wrong.** It is measured, but only on answer tokens. The held-out text domains, the benchmark passages and the tables all go the other way.

**The LAMBADA cost of reg35 is also an answer effect.** The final word gets worse (llama −9.7†, qwen −2.9), while the passage worsens only slightly (−1.3 / −0.6). That passage cost matches reg35's held-out Gutenberg and FineWeb losses.

## 3. Training-corpus format does not separate register from wiki

All five synthetic corpora come from the same FAQ rewrite prompt. 20,000 documents per corpus were profiled:

| | base | reg15 | reg35 | wiki15 | wiki35 | human FineWeb |
|---|---:|---:|---:|---:|---:|---:|
| FAQ documents (≥2 questions) | 82.5 | 79.8 | 76.4 | 78.5 | 74.9 | 10.7 |
| questions answered | 85.1 | 83.0 | 75.7 | 82.7 | 80.4 | – |
| compact `Q↵A:` layout (what the benchmarks use) | 43.6 | 40.4 | 33.1 | 40.5 | 38.4 | 1.5 |
| answer has an A-marker | 53.6 | 49.0 | 40.5 | 49.7 | 46.3 | – |
| bare question line (no marker) | 22.1 | 36.4 | 56.0 | 31.8 | 44.1 | – |
| answers ≤ 5 words | 1.9 | 2.9 | **6.3** | 3.0 | 5.7 | – |
| "how many / how much" questions | 1.8 | 2.2 | 2.8 | 2.2 | 2.7 | – |
| docs with a table | 0.4 | 0.4 | 0.3 | 0.6 | 0.6 | 2.9 |
| docs with a numbered list | 36.0 | 29.1 | 22.1 | 29.4 | 22.6 | 3.1 |

All values are %. Almost every difference vs base is significant at n = 20k; the full table with CIs is `outputs/format/FORMAT.md`.

- **Steering makes the FAQs *less* benchmark-like, and more so at α35, for both sources.** α35 corpora have fewer marked Q/A pairs, fewer compact `Q↵A:` pairs, more unanswered questions and more bare question lines.
- **Tables are practically absent** (≤ 0.6%) in every corpus, so table-QA gains can't come from seeing tables.
- **Register and wiki look almost the same at each α.** That matches the earlier result on frequency and predictability.
- **The features that rank reg35 first are short answers (6.3%) and slightly more "how many" / "who" questions.** These match the benchmark answer style (short spans and numbers in WikiTableQ and DROP), and short-answer share rises with α as answer gains do. But the reg35 − wiki35 gap is small (6.3 vs 5.7%), so format at this level does not explain why register leads wiki.

## 4. What this means for the hypothesis

- **Rejected:** "register steering produces benchmark-like formats or document genres". Corpus format moves *away* from the benchmark layout with steering, and register ≈ wiki.
- **Rejected:** "benchmark gains come from better modelling of tables or passages". Context gains are small, do not predict answer gains, and are *larger* for wiki.
- **Supported:** benchmark gains are a property of **answer prediction**. That is the distribution over the short span that follows a question.
  - Register (and α35) shifts that distribution towards the gold answers more than wiki does, even though it models ordinary text slightly worse.
  - This is the same trade-off seen in perplexity (α35 = worse prose, better on its special tokens), seen from the answer side.
- **Still open:** *what* in register text produces that. Two candidates fit the data but neither is tested:
  1. The register paraphrases pair questions with short, direct, extractive answers more often in content terms (the answer restates a span or entity from the source). Our line-based format features would miss this.
  2. A lexical shift in which tokens follow a question, such as numbers, names or the first word of a span.

  A next check would score answer-like spans in the training corpora (e.g. answer-token type distribution, or overlap between the answer and the source FineWeb document). It needs no training.

**Follow-up (done):** see [`FINDINGS_answer_content.md`](FINDINGS_answer_content.md).

## Caveats

- **Few-shot demos come from a fixed pool of 3–4 per task.** The `demos:` columns and the "gain by shot" table in `outputs/benchmark/SUMMARY.md` therefore reflect 3–4 specific answers. Their CIs are not meaningful, and I don't use them above.
- **DROP:** lighteval sampled different few-shot demos for each model. Here all models are scored on the base-llama prompt, so our DROP Δs are on identical prompts and can differ slightly from the published ones.
- **SQuAD:** only the first gold answer is scored.
- **Seeds:** as before, one seed per arm. The register − wiki answer contrasts (+3 to +8) are large relative to item noise, but the variation between seeds is unmeasured.
