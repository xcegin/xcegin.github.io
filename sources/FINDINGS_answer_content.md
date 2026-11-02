# What changes in the answers of the steered training data, and do the models pick it up?

Analyses run 2026-10-31 and 2026-11-01. The scripts and outputs are listed at the end.

## Summary

**Question.** The benchmark gains of the steered models come from giving more probability to the gold answer, not from better reading of the table or passage (background below). So what changes in the *answers* of the steered training data, and do the trained models behave differently when they have to answer?

**What we show:**
1. **At α35, answers in the steered training data more often open with a short, direct reply.**
   - Example: "What is her nationality? She's Nigerian." instead of a sentence that restates the question.
   - The answers as a whole get *longer*, not shorter. Only their opening changes.
2. **Steered answers contain more numbers that are not in the document they were rewritten from.**
   - The share roughly doubles from base to α35 (20% → 42–44% of answer numbers).
   - These are mostly added details, not copied facts.
3. **Register and wiki training data are almost the same on all of this.** The few differences are 1–3 percentage points.
4. **Every steered model prefers to answer with a short span** (a name, number or word) instead of a sentence, more than its base model. α35 models do this most. This holds with and without few-shot demos.
5. **This preference roughly follows the benchmark gains across the 8 models** (rank correlation 0.43–0.81, depending on the measure). There are exceptions, most clearly llama wiki35.

**What we don't show:**
- That the change in the training answers *causes* the span preference, or that the span preference causes the benchmark gains. Everything here is correlational, with one seed per model.
- What makes register better than wiki on benchmarks. It is not even certain that this difference is larger than seed noise.

## Background: what we are trying to explain

These earlier results motivate the analysis (details in [`FINDINGS_benchmark_and_format.md`](FINDINGS_benchmark_and_format.md)):

- **The benchmark gains are on the answer tokens.**
  - We scored all 10 models on the exact benchmark prompts. The loss on the gold answer improves by 5–25 centi-nats. Example: qwen reg35 on WikiTableQ, +24.9 on the answer vs +0.6 on the table and question.
  - Across items, how much better a model reads the context does not predict how much better it predicts the answer (Spearman ρ ≈ 0).
- **Register beats wiki on benchmarks in all 4 comparisons** (both architectures, both α). The margins are +0.12 to +0.38 points.
  - That is within the noise expected between two single-seed models (≈ 0.23–0.69, `model_results_analysis/LIGHTEVAL_NALL.md`).
  - So this difference may be partly or fully seed noise.
- **llama wiki_a35 has no benchmark gain over its base on the full benchmark suite** (macro +0.01). Any explanation has to account for that.
- **The format of the training corpora doesn't explain the gains.** Steering makes the FAQs *less* like the benchmark layout, and register ≈ wiki (`outputs/format/FORMAT.md`).

## Data and method

- **All 5 synthetic corpora rewrite the same FineWeb documents into FAQs.** Synthetic shard *i* rewrites FineWeb shard *i*, matched by document `id`.
- **We took 15,000 FineWeb documents that every corpus rewrote** (10 shards × 1,500).
  - From each rewrite we extracted the question → answer pairs.
  - We compared the rewrites of the *same* source document, so differences in topic or content between corpora don't affect the comparison.
- **Statistics:** uncertainty is a bootstrap over documents (or over benchmark items for the model tests). † means the 95% interval excludes 0. Percentages are shares of answers; differences are percentage points.
- **"Source document"** always means the FineWeb document a FAQ was rewritten from. The model never sees it during training; it only sees the FAQ.

## Finding 1: at α35 answers open more directly, but answers and documents get longer

Source: `outputs/answers/ANSWERS.md` and `outputs/format/FORMAT.md`.

| | base | reg15 | reg35 | wiki15 | wiki35 |
|---|---:|---:|---:|---:|---:|
| words per document | 320 | 337 | 416 | 314 | 390 |
| Q/A pairs per document | 5.8 | 6.1 | 6.9 | 6.0 | 6.7 |
| words per answer | 43.6 | 44.7 | 54.0 | 42.5 | 50.7 |
| answer opens with a short sentence (≤ 8 words) | 13.7% | 18.3% | **24.8%** | 17.9% | **23.4%** |
| answer first restates the question | 44.8% | 44.5% | 39.0% | 43.3% | 38.7% |
| answer contains a number | 30.6% | 31.4% | 33.5% | 31.0% | 33.0% |
| whole answer ≤ 3 words | 0.9% | 1.8% | 3.9% | 1.7% | 3.7% |

All α35 − base differences in the first six rows are †. The last row is pooled over all answers, without an interval.

- **"Shorter answers" is the wrong description.** At α35, documents are longer because they have more Q/A pairs, and each answer is longer too.
- **What changes is how an answer starts.** It more often opens with a short direct reply and less often restates the question. Example from reg35: *"What kind of song is Pareke? It's a street-ready song, Teni says."*
- **This moves the answers only a little towards benchmark gold answers.** Gold answers are 1–2 words long (median); 72–96% of them are ≤ 3 words, depending on the task. Only about 4% of α35 training answers are that short.

## Finding 2: steered answers contain more numbers that are not in the source document

Source: `outputs/answers/NUMBERS.md`.

**How it was measured.** For every number in an answer, we checked whether it occurs in the source FineWeb document. If not, we assigned the first reason that applies:
- list or step numbering ("1.", "Step 2");
- the source writes the number as a word ("2" ↔ "two");
- a reformatted source number ("1,500" vs "1500");
- the number comes from the question itself.

Anything left is "not supported".

| corpus | in source | list/step numbering | word form / reformatted / from question | **not supported** | answers with ≥ 1 unsupported number |
|---|---:|---:|---:|---:|---:|
| base | 63.5% | 10.9% | 5.9% | **19.6%** | 7.2% |
| reg15 | 54.3% | 11.5% | 7.0% | **27.1%** | 10.3% |
| reg35 | 37.6% | 10.4% | 10.2% | **41.9%** | 16.3% |
| wiki15 | 55.0% | 10.1% | 7.1% | **27.7%** | 10.1% |
| wiki35 | 38.5% | 7.5% | 10.2% | **43.7%** | 16.5% |

Shares of all numbers in answers: 61k–100k numbers per corpus.

- **The share of unsupported numbers roughly doubles from base to α35,** and is the same for register and wiki.
- **What these numbers are** (from reading the examples in `NUMBERS.md`):
  - Mostly details the rewrite added, e.g. "increase house-building output to 200,000", "teaching since 1981", "expected to make an impact by 2018". None of these are in the source.
  - A few are true background knowledge (e.g. "the 2016 DNC email leaks").
  - A few are misses of the classifier (list numbering written as "1)", digits inside product names).
- **So "less faithful" means precisely:** more of the numbers in the answers were added by the rewrite instead of taken from the source. Most are made up, but not all are false.
- **The rewrites also keep fewer of the source's numbers** (40.8% in base vs 32% at α35) **and fewer of its content words** (32.8% vs 26–28%).

## Finding 3: register and wiki training data differ only slightly

Source: `outputs/answers/ANSWERS.md`, register vs wiki section.

Paired differences on the same source documents, in percentage points:

| | reg15 − wiki15 | reg35 − wiki35 |
|---|---:|---:|
| answer starts with "yes"/"no" | −1.4† | −3.1† |
| answer starts with a content word (not "The", "It", …) | +0.9† | +2.2† |
| answer opens with a short sentence (≤ 8 words) | +0.5 | +1.8† |
| answer starts with words or a number from the source | +1.2† | +1.7† |
| answer numbers found in the source (per-document average) | +0.4 | +0.2 |

- **Register answers are a little closer to benchmark answers.** Gold answers almost never start with yes/no (≤ 1.4%) and mostly start with a name, number or content word.
- **The differences are 1–3 points on base rates of 10–60%.** They are consistent in direction, but too small to show that they explain register's benchmark lead.
- **On faithfulness, register and wiki are the same.**

## Finding 4: the steered models prefer to answer with a short span

Source: `outputs/preference/PREFERENCE.md`.

**How it was measured.**
- **Items:** 2,500 benchmark items (500 each from WikiTableQ, TREB-QA, SQuADv2, DROP and TriviaQA). For each model we looked at what it does right after `Answer:`.
- **Span vs sentence:** the probability of the first token being a *span* start (a number, name or other word) vs a *sentence* start ("The", "It", "This", "According", "Yes", …).
- **Greedy answer:** the model's own answer, up to the first line break, counted as span-like (≤ 3 words, not starting with a sentence opener) or sentence-like.
- **Two settings:**
  - **0-shot:** only the question and its passage or table, no demos. This shows the model's own habit.
  - **3-shot:** the exact benchmark prompt, where three demos with short answers come first.

**Results** (percentage points vs the family's base model; † = 95% interval excludes 0):

| model | P(span), 0-shot | span-like greedy answer, 0-shot | P(span), 3-shot | span-like greedy answer, 3-shot |
|---|---:|---:|---:|---:|
| llama reg15 | +2.8† | +1.0 | +2.2† | +3.6† |
| llama reg35 | +8.8† | +14.1† | +4.9† | +6.6† |
| llama wiki15 | +1.7† | +1.4† | +2.3† | +2.9† |
| llama wiki35 | +5.3† | +9.1† | +1.7† | +3.3† |
| qwen reg15 | +3.9† | +5.3† | +2.9† | +7.6† |
| qwen reg35 | +7.6† | +11.6† | +4.4† | +9.7† |
| qwen wiki15 | +0.5† | +5.9† | +3.0† | +9.4† |
| qwen wiki35 | +6.9† | +13.5† | +5.7† | +12.0† |

Base models' levels:
- P(span) is 49.6% / 49.8% at 0-shot and 77.1% / 77.2% at 3-shot (llama / qwen).
- Greedy answers are sentence-like 88.9% / 91.1% of the time at 0-shot.

- **Every steered model prefers spans more than its base, in both settings.** 31 of 32 differences are †; the exception is llama reg15's 0-shot greedy answers.
- **The effect is strongest at α35 and without demos.**
  - At 0-shot, base models answer with a sentence about 90% of the time; reg35 and wiki35 do so 74–79% of the time.
  - With 3 demos, every model already prefers spans (base 77%), so the differences shrink to +2 to +6 points in P(span).
- **Register vs wiki depends on the setting:**
  - **0-shot:** register prefers spans more than wiki in all 4 comparisons: llama +2.8 vs +1.7 and +8.8 vs +5.3; qwen +3.9 vs +0.5 and +7.6 vs +6.9.
  - **3-shot (the benchmark setting):** register and wiki are about equal at α15 (llama 2.2 vs 2.3, qwen 2.9 vs 3.0). Register is ahead at llama α35 (4.9 vs 1.7), but wiki is ahead at qwen α35 (5.7 vs 4.4).
  - So the model-level register advantage is clear only without demos.
- **The models do not produce the correct answer more often.** Greedy answers contain the gold answer 15–19% of the time for every model, in both settings. The change is in the *form* of the answer and in how much probability the gold answer gets, which is what these benchmarks score.

## Finding 5: the span preference roughly follows the benchmark gains, with exceptions

Benchmark gain here is the mean Δ prob_norm_token ×100 on the same 5 QA tasks, from step 6. Rank correlation (Spearman) across the 8 steered models, between the preference shift (Finding 4) and the benchmark gain:

| preference measure | 0-shot | 3-shot |
|---|---:|---:|
| P(span) | 0.81 | 0.67 |
| span-like greedy answer | 0.57 | 0.43 |

With only 8 models, these correlations are descriptive.

| | llama reg15 | llama reg35 | llama wiki15 | llama wiki35 | qwen reg15 | qwen reg35 | qwen wiki15 | qwen wiki35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benchmark gain (5 QA tasks) | +0.94 | +1.58 | +1.15 | +0.61 | +1.30 | +1.72 | +0.31 | +1.44 |
| P(span) shift, 0-shot | +2.8 | +8.8 | +1.7 | +5.3 | +3.9 | +7.6 | +0.5 | +6.9 |

- **The models with the largest span shift (reg35, qwen wiki35) also have the largest benchmark gains.** At 0-shot, qwen's four models rank in the same order on both.
- **llama wiki35 doesn't fit at 0-shot:** the second-largest span shift in llama (+5.3), but the smallest benchmark gain (+0.61; and +0.01 on the full suite). At 3-shot its span shift is the smallest in llama (+1.7), which does fit.
- **llama wiki15 has a larger benchmark gain than llama reg15 with a smaller 0-shot span shift.**
- **The preference is therefore one plausible part of the benchmark gains, not a full explanation.**

## Finding 6: register's answer advantage is spread over names and numbers

Source: `outputs/answers/ANSWER_TYPES.md`.

Gold answers were grouped by their first word. Register − wiki at α35, gold-answer loss gain pooled over the 5 QA tasks (centi-nats, 95% interval):

| | llama | qwen |
|---|---:|---:|
| answers that are numbers (n = 2,338) | +4.0 [+3.0, +4.9] | +2.9 [+2.1, +3.7] |
| answers that are names (n = 4,202) | +6.7 [+4.8, +8.7] | +3.2 [+1.3, +5.1] |

- **The advantage is not limited to one answer type.** That argues against a narrow skill like "copies numbers better", and fits a general preference for short, specific answers.

## What we show, what we don't, and the evidence

| Statement | Status | Evidence |
|---|---|---|
| Benchmark gains are on the gold answer, not on reading the context | shown | answer gains of 5–25 vs context gains of ~1 centi-nat; ρ ≈ 0 across items (`FINDINGS_benchmark_and_format.md`) |
| At α35, training answers open more directly (but are longer overall) | shown | Finding 1: short opening sentence 13.7% → 23–25%; answers 44 → 51–54 words |
| At α35, training answers contain more numbers not in the source | shown | Finding 2: 19.6% → 42–44%, after removing list numbering and reformatting |
| Register and wiki training data are nearly identical | shown | Findings 2–3 and `FORMAT.md`: differences of 0–3 points |
| Register answers are slightly closer in form to benchmark answers | shown, small | Finding 3: −1.4/−3.1 yes/no starts, +0.9/+2.2 content-word starts |
| Steered models prefer span answers more than their base | shown, 0-shot and 3-shot | Finding 4: 31 of 32 differences †, largest at α35 |
| Register models prefer spans more than wiki models | only 0-shot | Finding 4: 4 of 4 comparisons at 0-shot, mixed at 3-shot |
| The span preference follows the benchmark gains | partly | Finding 5: ρ = 0.43–0.81; llama wiki35 and wiki15 don't fit |
| The changed training answers *cause* the span preference | **not shown** | correlational; one seed per model; needs training experiments |
| The span preference *causes* the benchmark gains | **not shown** | correlational; the benchmark rewards gold-answer likelihood, which a span preference would raise, but this is not tested |
| What makes register better than wiki on benchmarks | **not known** | corpus differences are small; the benchmark margin may be seed noise |

**Current working explanation, as a hypothesis.**
1. Steering, most strongly at α35, makes FAQ answers open with a short, specific reply, often with added details.
2. Models trained on these answers prefer short span answers after a question.
3. Benchmarks that score the probability of a short gold answer reward that.

The same models do slightly worse on ordinary prose (earlier perplexity results). The steps between these observations are untested.

**What would test it:** extra seeds for one register and one wiki model per architecture (is register vs wiki real?), the α25 run, and the temperature-matched baseline (is it the steering direction, or just more varied text?).

## Caveats

- **The source document is never in the model's training context.** "Not in the source" measures what the rewrite added, not something the model could copy. The FAQs contain almost no passages, so we can't measure copying from earlier in the same document.
- **The answer extraction is line-based.**
  - A line is a question if it has a question marker ("Q:", "Q1.", "Question:"), is a numbered line or heading ending in "?", or is a line of ≤ 40 words ending in "?".
  - The answer is everything up to the next question.
- **The span/sentence split uses a fixed opener list** ("The", "It", "This", "According", "Yes", …). A different list would shift the levels, but not the comparison between models, which all use the same list.
- **The α35 mixes also have a different human half** (other CommonCrawl dumps; `RUN_PLAN.md`), which is collinear with α. Its measured effect on predictability is small, but it is not ruled out as a cause of the model differences.
- **One seed per model.** For the model-level results, the intervals cover item noise only, not seed noise.

## Scripts and outputs

| Step | Script | Output |
|---|---|---|
| answer content of the corpora, paired | `scripts/answer_content.py` via `08_answer_content.sbatch` (job 210336) | `outputs/answers/ANSWERS.md`, `answer_content.csv`, `answer_contrasts.csv` |
| benchmark gains by answer type | `scripts/benchmark_answer_types.py` via `08_answer_content.sbatch` | `outputs/answers/ANSWER_TYPES.md` |
| numbers not in the source | `scripts/number_faithfulness.py` via `09_answer_preference.sbatch` (job 210392) | `outputs/answers/NUMBERS.md` |
| span preference of the models | `scripts/answer_preference.py` via `09_answer_preference.sbatch` | `outputs/preference/PREFERENCE.md`, `preference_vs_base.csv` |

Submit `09` with `--exclude=gn010`: that node's GPU has failed CUDA initialisation since 2026-11-01.
