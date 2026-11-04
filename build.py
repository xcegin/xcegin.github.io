#!/usr/bin/env python3
"""
Build page_report/index.html from the CSVs in page_report/data/ (copied from model_results_analysis/ and
loss_attribution_analysis/). Standard library only:  python3 build.py
Every number on the page is computed here from those files; prose that is not computed cites its source file.
"""
import csv
import html
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
ARMS = ["reg15", "reg35", "wiki15", "wiki35"]
LONG = {"reg15": "register_a15", "reg35": "register_a35", "wiki15": "wiki_a15", "wiki35": "wiki_a35"}
NICE = {"reg15": "register α15", "reg35": "register α35", "wiki15": "wiki α15", "wiki35": "wiki α35"}
FAM = {"llama": "Llama 1B", "qwen": "Qwen2 1.7B"}
FAMDIR = {"llama": "llama_1b", "qwen": "qwen2_1.7b"}
# benchmark families (qwen 2.9B has only base, register α35, wiki α35); the loss/data sections cover FAM only
BFAM = {"llama": ("Llama 1B", ARMS), "qwen": ("Qwen2 1.7B", ARMS), "qwen29": ("Qwen2 2.9B", ["reg35", "wiki35"])}
BKEYS = [(f, a) for f in BFAM for a in BFAM[f][1]]
MC = ["XCSQA", "HellaSwag", "ARC-Easy", "WinoGrande", "PIQA", "ARC-Challenge", "OpenBookQA", "MMLU-Redux"]


def rd(name):
    with open(os.path.join(D, name), newline="") as f:
        return list(csv.DictReader(f))


def esc(s):
    return html.escape(str(s))


def fmt(v, nd=2, sign=True):
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


# ------------------------------------------------------------------ data
CHECKS = []          # (description, value) printed by the build for double-checking


def lighteval():
    out = {}
    for fam in BFAM:
        agg = {r["model"]: r for r in rd(f"lighteval_{fam}_aggregates.csv") if r["aggregate"] == "agg_macro"}
        out[fam] = {a: (100 * float(agg[LONG[a]]["delta"]), 100 * float(agg[LONG[a]]["ci_lo"]),
                        100 * float(agg[LONG[a]]["ci_hi"])) for a in BFAM[fam][1]}
        out[fam]["base"] = 100 * float(agg["base"]["value"])
    return out


def lighteval_contrasts():
    out = {}
    for fam in BFAM:
        for r in rd(f"lighteval_{fam}_contrasts.csv"):
            if r["aggregate"] == "agg_macro":
                out[(fam, r["contrast"])] = (100 * float(r["delta"]), 100 * float(r["ci_lo"]), 100 * float(r["ci_hi"]))
    return out


TASKS = ["WikiTableQ", "TREB-QA", "DROP", "SQuADv2", "TriviaQA", "XCSQA", "ARC-Easy", "ARC-Challenge", "OpenBookQA",
         "HellaSwag", "PIQA", "WinoGrande", "MMLU-Redux", "GSM8K", "LAMBADA"]


def task_deltas():
    out, acc = {}, {}
    inv = {v: k for k, v in LONG.items()}
    for fam in BFAM:
        for r in rd(f"lighteval_{fam}_task_deltas.csv"):
            k = (fam, inv[r["model"]], r["name"])
            if r["metric"] == "prob_norm_token":
                out[k] = (100 * float(r["delta"]), float(r["q_bh"]))
            elif r["metric"] == "acc_norm_token":
                acc[k] = (100 * float(r["delta"]), float(r["q_bh"]), 100 * float(r["base"]))
    return out, acc


DOMS = [("fineweb", "FineWeb"), ("wikipedia", "Wikipedia"), ("arxiv", "arXiv"), ("hack", "HackerNews"),
        ("stack", "StackExchange"), ("freelaw", "FreeLaw"), ("gutenberg", "Gutenberg"), ("MACRO", "Mean")]


def ppl():
    out = {}
    fam_of = {"llama_1b": "llama", "qwen2_1.7b": "qwen"}
    inv = {v: k for k, v in LONG.items()}
    for r in rd("ppl_deltas.csv"):
        if r["variant"] == "all":
            out[(fam_of[r["family"]], inv[r["model"]], r["domain"])] = (100 * float(r["delta"]), float(r["q_bh"]),
                                                                         100 * float(r["lo"]), 100 * float(r["hi"]))
    return out


def buckets():
    want = [("difficulty (quintile of mean loss)", "easiest", "easiest 20% of tokens"),
            ("difficulty (quintile of mean loss)", "hardest", "hardest 20% of tokens"),
            ("frequency rank (training corpus)", "unseen", "tokens unseen in base training corpus"),
            ("frequency rank (training corpus)", "top-100", "100 most frequent tokens"),
            ("token class", "function", "function words"),
            ("token class", "whitespace", "whitespace / layout")]
    out = {}
    for fam in FAM:
        R = rd(f"attribution_{fam}.csv")
        for a in ARMS:
            for an, b, lab in want:
                v = [float(r["mean_delta"]) for r in R if r["analysis"] == an and r["bucket"] == b and r["arm"] == f"{fam}_{a}"]
                out[(fam, a, lab)] = 100 * st.median(v)
    return out, [w[2] for w in want]


def words_vs_ws():
    """Δ on non-whitespace tokens, derived: total = s*ws + (1-s)*words  ->  words = (total - s*ws)/(1-s)."""
    tot = {(r["domain"], r["family"], r["condition"]): float(r["delta"]) for r in rd("token_overall.csv")}
    out = {}
    for r in rd("token_special.csv"):
        if r["token"] == "all whitespace":
            k = (r["domain"], r["family"], r["condition"])
            s, ws = float(r["token_share"]), float(r["mean_delta"])
            out[k] = (100 * (tot[k] - s * ws) / (1 - s), 100 * ws)
    return out


def calib():
    out = {}
    for r in rd("token_calib_quintiles.csv"):
        if r["quintile"] == "hardest":
            out.setdefault((r["family"], r["condition"]), []).append((float(r["d_nll"]), float(r["d_entropy"])))
    return {k: (100 * st.median([x[0] for x in v]), 100 * st.median([x[1] for x in v])) for k, v in out.items()}


def corpus_pred():
    out = {}
    for sc in ("qwen", "pythia"):
        for r in rd(f"corpus_predictability_{sc}.csv"):
            out[(sc, r["arm"], r["group"])] = (float(r["delta"]), float(r["lo"]), float(r["hi"]))
    return out


def bench():
    A = rd("bench_answer.csv")
    S = rd("bench_segments.csv")
    out = {}
    for fam in FAM:
        for a in ARMS:
            ans = [float(r["delta_ans"]) for r in A if r["family"] == fam and r["arm"] == a and r["task"] != "LAMBADA"]
            ctx = [float(r["delta"]) for r in S if r["family"] == fam and r["arm"] == a and r["task"] != "LAMBADA"
                   and r["segment"] == "target:context"]
            out[(fam, a)] = (st.mean(ans), st.mean(ctx), len(ans))
    per_task = {(r["family"], r["arm"], r["task"]): float(r["delta_ans"]) for r in A}
    ctx_task = {(r["family"], r["arm"], r["task"]): float(r["delta"]) for r in S if r["segment"] == "target:context"}
    link = rd("bench_link.csv")
    rho = [abs(float(r["spearman"])) for r in link]
    con = {}
    for r in rd("bench_contrasts.csv"):
        if r["segment"] in ("answer", "target:context"):
            con[(r["family"], r["contrast"], r["task"], r["segment"])] = (float(r["delta"]), float(r["q_bh"]))
    san = rd("bench_sanity.csv")
    corr = [float(r["corr_lp"]) for r in san if r["corr_lp"]]
    return out, per_task, ctx_task, max(rho), con, min(corr)


def answers():
    C = {(r["feature"], r["corpus"]): float(r["value"]) for r in rd("answers_content.csv")}
    K = {(r["feature"], r["contrast"]): (float(r["diff"]), float(r["lo"]), float(r["hi"])) for r in rd("answers_contrasts.csv")}
    N = {r["corpus"]: r for r in rd("answers_numbers.csv")}
    F = {(r["corpus"], r["feature"]): r for r in rd("corpus_format.csv")}
    return C, K, N, F


def preference():
    P = {(r["cond"], r["arm"]): r for r in rd("preference.csv")}
    V = {(r["cond"], r["arm"], r["metric"]): (100 * float(r["diff"]), 100 * float(r["lo"]), 100 * float(r["hi"]))
         for r in rd("preference_vs_base.csv")}
    return P, V


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


# ------------------------------------------------------------------ svg helpers
def hbars(rows, xmin, xmax, unit, band=None, width=640, label_w=200, row_h=30, series=None, legend=None, title=""):
    """rows: list of (label, [ (value, lo, hi, series_idx), ... ])  -> horizontal bar chart (one or two bars per row)."""
    nser = max(len(v) for _, v in rows)
    bar_h = 10 if nser > 1 else 14
    top, bottom = 10, 34
    h = top + bottom + row_h * len(rows)
    plot_w = width - label_w - 60
    X = lambda v: label_w + (v - xmin) / (xmax - xmin) * plot_w
    s = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}" class="chart">']
    if band:
        s.append(f'<rect x="{X(band[0]):.1f}" y="{top}" width="{X(band[1]) - X(band[0]):.1f}" height="{row_h * len(rows)}" '
                 f'class="band"><title>{esc(band[2])}</title></rect>')
    # grid + axis
    step = nice_step(xmax - xmin)
    t = (int(xmin / step) - 1) * step
    while t <= xmax + 1e-9:
        if t >= xmin - 1e-9:
            s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + row_h * len(rows)}" class="{"zero" if abs(t) < 1e-9 else "grid"}"/>')
            s.append(f'<text x="{X(t):.1f}" y="{h - 18}" class="tick" text-anchor="middle">{t:g}</text>')
        t += step
    s.append(f'<text x="{label_w + plot_w / 2:.1f}" y="{h - 2}" class="axis" text-anchor="middle">{esc(unit)}</text>')
    for i, (lab, vals) in enumerate(rows):
        y0 = top + i * row_h
        s.append(f'<text x="{label_w - 8}" y="{y0 + row_h / 2 + 4:.1f}" class="lab" text-anchor="end">{esc(lab)}</text>')
        for j, (v, lo, hi, si) in enumerate(vals):
            yb = y0 + (row_h - bar_h * len(vals) - 2 * (len(vals) - 1)) / 2 + j * (bar_h + 2)
            x0, x1 = sorted((X(0), X(v)))
            name = (series[si] + ": ") if series else ""
            tip = f"{lab} — {name}{v:+.2f}" + (f" [{lo:+.2f}, {hi:+.2f}]" if lo is not None else "")
            s.append(f'<g class="mark"><title>{esc(tip)}</title>'
                     f'<rect x="{x0:.1f}" y="{yb:.1f}" width="{max(x1 - x0, 1):.1f}" height="{bar_h}" rx="2" class="s{si}"/>')
            if lo is not None:
                s.append(f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{yb + bar_h / 2:.1f}" y2="{yb + bar_h / 2:.1f}" class="ci"/>')
            tx = X(max(v, hi if hi is not None else v)) + 5 if v >= 0 else X(min(v, lo if lo is not None else v)) - 5
            s.append(f'<text x="{tx:.1f}" y="{yb + bar_h - 1:.1f}" class="val" text-anchor="{"start" if v >= 0 else "end"}">{v:+.2f}</text></g>')
    s.append("</svg>")
    leg = ""
    if legend:
        leg = '<div class="legend">' + "".join(f'<span><i class="sw s{k}"></i>{esc(n)}</span>' for k, n in enumerate(legend)) + "</div>"
    return leg + "".join(s)


def nice_step(r):
    for s in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50):
        if r / s <= 8:
            return s
    return 100


def heat(v, vmax):
    """diverging cell background via color-mix: blue = better than base, red = worse."""
    a = min(abs(v) / vmax, 1.0) * 70
    col = "var(--pos)" if v > 0 else "var(--neg)"
    return f"background:color-mix(in oklab, {col} {a:.0f}%, var(--mid))"


def heat_table(cols, rows, vmax, note=""):
    """rows: (row label, [(value, significant_bool), ...])"""
    h = ['<div class="scroll"><table class="heat"><thead><tr><th></th>' + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr></thead><tbody>"]
    for lab, cells in rows:
        h.append(f"<tr><th>{esc(lab)}</th>" + "".join(
            f'<td style="{heat(v, vmax)}" title="{esc(lab)}, {esc(c)}: {v:+.2f}{" (q<0.05)" if sig else ""}">{v:+.2f}{"<sup>†</sup>" if sig else ""}</td>'
            for c, (v, sig) in zip(cols, cells)) + "</tr>")
    h.append("</tbody></table></div>")
    if note:
        h.append(f'<p class="note">{note}</p>')
    return "".join(h)


def table(head, rows, cls="tbl"):
    return ('<div class="scroll"><table class="' + cls + '"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) +
            "</tbody></table></div>")


# ------------------------------------------------------------------ build
def main():
    global CHECKS
    LE, LC, PP = lighteval(), lighteval_contrasts(), ppl()
    TD, ACC = task_deltas()
    BK, bucket_labels = buckets()
    WW, CAL, CP = words_vs_ws(), calib(), corpus_pred()
    BE, BT, BC, rho_max, BCON, san_corr = bench()
    AC, AK, AN, AF = answers()
    PR, PV = preference()

    sec = []

    # ---------- TL;DR numbers
    best = max(((LE[f][a][0], f, a) for f, a in BKEYS))
    BN = {f: BFAM[f][0] for f in BFAM}
    CHECKS.append(("best macro", best))

    # ---------- 1. benchmarks
    rows = [(f"{BN[f]} · {NICE[a]}", [(LE[f][a][0], LE[f][a][1], LE[f][a][2], 0)]) for f, a in BKEYS]
    fig1 = hbars(rows, -0.2, 1.2, "Δ macro score vs base (points)", band=(0.16, 0.49, "seed-to-seed SD of this score: 0.16–0.49 points (finephrase paper, Table 21)"),
                 title="Benchmark macro score change vs base")
    con_rows = []
    for name, key in [("register − wiki, α15", "register_a15 - wiki_a15"), ("register − wiki, α35", "register_a35 - wiki_a35"),
                      ("α35 − α15, register", "register_a35 - register_a15"), ("α35 − α15, wiki", "wiki_a35 - wiki_a15")]:
        con_rows.append([name] + [f"{LC[(f, key)][0]:+.2f} [{LC[(f, key)][1]:+.2f}, {LC[(f, key)][2]:+.2f}]"
                                  if (f, key) in LC else "–" for f in BFAM])
    hrows = [(f"{BN[f]} · {NICE[a]}", [(TD[(f, a, t)][0], TD[(f, a, t)][1] < 0.05) for t in TASKS]) for f, a in BKEYS]
    fig2 = heat_table(TASKS, hrows, 3.0, "Δ prob_norm_token × 100 vs base, full evaluation sets. † = significant after BH correction (q &lt; 0.05). "
                      "Blue = better than base, red = worse. Hover a cell for its value.")
    n_sig_pos = sum(1 for f, a in BKEYS for t in TASKS if TD[(f, a, t)][1] < 0.05 and TD[(f, a, t)][0] > 0)
    n_sig_neg = sum(1 for f, a in BKEYS for t in TASKS if TD[(f, a, t)][1] < 0.05 and TD[(f, a, t)][0] < 0)
    wtq = [TD[(f, a, "WikiTableQ")][0] for f, a in BKEYS]
    drop = [TD[(f, a, "DROP")][0] for f, a in BKEYS]
    xcs = [TD[(f, a, "XCSQA")][0] for f, a in BKEYS]
    n_drop = sum(TD[(f, a, "DROP")][0] > 0 and TD[(f, a, "DROP")][1] < 0.05 for f, a in BKEYS)
    n_wtq = sum(TD[(f, a, "WikiTableQ")][0] > 0 and TD[(f, a, "WikiTableQ")][1] < 0.05 for f, a in BKEYS)
    # probability vs accuracy (multiple-choice tasks)
    acc_rows = []
    for t in MC:
        acc_rows.append([f"{t} <span class='note'>({' / '.join(f'{ACC[(f, BFAM[f][1][0], t)][2]:.0f}' for f in BFAM)})</span>"] +
                        [f"{ACC[(f, a, t)][0]:+.1f}{'†' if ACC[(f, a, t)][1] < 0.05 else ''} / {TD[(f, a, t)][0]:+.2f}{'†' if TD[(f, a, t)][1] < 0.05 else ''}"
                         for f, a in BKEYS])
    def nsig(D_, f, a, sign):
        return sum(1 for t in MC if D_[(f, a, t)][1] < 0.05 and sign * D_[(f, a, t)][0] > 0)
    acc_rows.append(["<b>significant gains / losses, probability</b>"] + [f"{nsig(TD, f, a, 1)} / {nsig(TD, f, a, -1)}" for f, a in BKEYS])
    acc_rows.append(["<b>significant gains / losses, accuracy</b>"] + [f"{nsig(ACC, f, a, 1)} / {nsig(ACC, f, a, -1)}" for f, a in BKEYS])
    short = {"llama": "L", "qwen": "Q1.7", "qwen29": "Q2.9"}
    acc_head = ["task <span class='note'>(base acc %: L / Q1.7 / Q2.9)</span>"] + [f"{short[f]} {NICE[a].replace('register', 'reg')}" for f, a in BKEYS]
    small_p = [nsig(TD, f, a, 1) for f in ("llama", "qwen") for a in ARMS]
    small_a = [nsig(ACC, f, a, 1) for f in ("llama", "qwen") for a in ARMS]
    CHECKS += [("acc reg35 2.9B", {t: ACC[("qwen29", "reg35", t)][:2] for t in MC})]
    CHECKS += [("sig pos/neg task deltas", (n_sig_pos, n_sig_neg)), ("WTQ range", (min(wtq), max(wtq))), ("DROP", drop),
               ("LAMBADA reg35", [TD[(f, "reg35", "LAMBADA")] for f in BFAM]),
               ("GSM8K llama", [TD[("llama", a, "GSM8K")] for a in ARMS])]
    sec.append(f"""
<section id="benchmarks"><h2>1 · Benchmarks: small gains; one clearly above seed noise</h2>
<p>Paper-style evaluation (lighteval, 3-shot, cloze, <code>prob_norm_token</code>), every question of every task, paired bootstrap against each architecture's base. Shaded band: the finephrase paper's seed-to-seed SD of this score (0.16–0.49 points). One seed per model.</p>
{fig1}
<ul>
<li>Largest gain: <b>{BN[best[1]]} {NICE[best[2]]} {best[0]:+.2f}</b> [{LE[best[1]][best[2]][1]:+.2f}, {LE[best[1]][best[2]][2]:+.2f}]. Qwen2 2.9B register α35 {LE['qwen29']['reg35'][0]:+.2f}, wiki α35 {LE['qwen29']['wiki35'][0]:+.2f}. Base macro: Llama {LE['llama']['base']:.2f}, Qwen 1.7B {LE['qwen']['base']:.2f}, Qwen 2.9B {LE['qwen29']['base']:.2f}.</li>
<li>The 2.9B models saw the same 20B tokens (≈7.5 tokens/parameter vs ≈12 at 1.7B), so they are the least trained.</li>
</ul>
<h3>Direct comparisons (macro Δ, points, 95% CI)</h3>
{table(['contrast'] + [BN[f] for f in BFAM], con_rows)}
<p class="note">Register beats wiki in all five comparisons (widest at 2.9B: {LC[('qwen29', 'register_a35 - wiki_a35')][0]:+.2f}); single-seed margins below ~0.23–0.69 are within the noise expected between two models.</p>
<h3>Per task</h3>
{fig2}
<ul>
<li>{n_sig_pos} of {len(BKEYS) * len(TASKS)} model × task changes are significantly positive, {n_sig_neg} significantly negative.</li>
<li><b>Gains:</b> DROP (significant in {n_drop} of {len(BKEYS)} models, up to {max(drop):+.2f}), WikiTableQ ({n_wtq} of {len(BKEYS)}; {min(wtq):+.2f} to {max(wtq):+.2f}), XCSQA (positive in all {len(BKEYS)}, up to {max(xcs):+.2f}), GSM8K at both Qwen sizes, TriviaQA at 2.9B.</li>
<li><b>Costs:</b> LAMBADA for register α35 in all three families ({', '.join(f"{TD[(f, 'reg35', 'LAMBADA')][0]:+.2f}" for f in BFAM)}), SQuADv2 at 2.9B, GSM8K on Llama (down to {min(TD[('llama', a, 'GSM8K')][0] for a in ARMS):+.2f}). Knowledge tasks (MMLU-Redux, ARC-Challenge, OpenBookQA) barely move.</li>
</ul>
<h3>Probability vs accuracy: what the models actually learn</h3>
<p>Only the 8 multiple-choice tasks have an accuracy (<code>acc_norm_token</code>). The two metrics answer different questions: <b>probability</b> = how much mass the correct answer gets; <b>accuracy</b> = whether it beats the wrong options. Probability up with accuracy flat means the wrong options rose too (a general boost to answer-like text). Accuracy up with probability flat means better choices without more confidence.</p>
{table(acc_head, acc_rows)}
<p class="note">Accuracy Δ / probability Δ, points; † = BH q &lt; 0.05. Accuracy needs ~2–3 points to be significant on these set sizes (500–2,400 questions; HellaSwag 10,042).</p>
<ul>
<li><b>Llama 1B and Qwen 1.7B: mostly a format effect.</b> Probability rises on {min(small_p)}–{max(small_p)} of 8 tasks, accuracy on only {min(small_a)}–{max(small_a)} (XCSQA, HellaSwag).</li>
<li><b>Qwen 2.9B: better choices.</b> Probability is flat, but register α35 accuracy rises on ARC-Easy ({ACC[('qwen29', 'reg35', 'ARC-Easy')][0]:+.1f}), XCSQA ({ACC[('qwen29', 'reg35', 'XCSQA')][0]:+.1f}) and HellaSwag ({ACC[('qwen29', 'reg35', 'HellaSwag')][0]:+.1f}, while its probability falls).</li>
<li><b>The prose cost is lost confidence, not worse judgement</b> (HellaSwag at 2.9B); WinoGrande at 2.9B is the only task where both point down.</li>
<li><b>In short:</b> better at turning a question into a short, specific answer; worse at plain-prose likelihood and exact span copying; knowledge unchanged. The size pattern rests on one seed per size.</li>
</ul>
<p class="src">Sources: <a href="data/lighteval_llama_aggregates.csv">aggregates</a>, <a href="data/lighteval_llama_task_deltas.csv">task deltas</a>, <a href="data/lighteval_llama_contrasts.csv">contrasts</a> (and the <code>qwen</code>, <code>qwen29</code> files); write-up <a href="sources/LIGHTEVAL_NALL.md">LIGHTEVAL_NALL.md</a> (findings 6–7).</p>
</section>""")

    # ---------- 2. perplexity
    prows = []
    for f in FAM:
        for a in ARMS:
            prows.append((f"{FAM[f]} · {NICE[a]}", [(PP[(f, a, d)][0], PP[(f, a, d)][1] < 0.05) for d, _ in DOMS]))
    fig3 = heat_table([n for _, n in DOMS], prows, 3.0, "Δ loss vs base in centi-nats (nats × 100) on the full held-out splits; positive = lower loss than base. † = BH q &lt; 0.05.")
    pc = {(r["family"], r["contrast"], r["domain"]): float(r["delta"]) * 100 for r in rd("ppl_contrasts.csv") if r["variant"] == "all"}
    fw35 = [PP[(f, a, "fineweb")][0] for f in FAM for a in ("reg35", "wiki35")]
    CHECKS.append(("fineweb a35", fw35))
    sec.append(f"""
<section id="perplexity"><h2>2 · Held-out loss: every model improves, wiki most, α35 least</h2>
<p>Per-domain held-out loss, recomputed on the full evaluation splits (the <code>final_eval.json</code> written at the end of training held the FineWeb loss under every domain name). Paired bootstrap over sequences.</p>
{fig3}
<ul>
<li>Mean gain: wiki α15 {PP[('llama', 'wiki15', 'MACRO')][0]:+.2f} / {PP[('qwen', 'wiki15', 'MACRO')][0]:+.2f} (Llama / Qwen) down to register α35 {PP[('llama', 'reg35', 'MACRO')][0]:+.2f} / {PP[('qwen', 'reg35', 'MACRO')][0]:+.2f} centi-nats, i.e. about 1–2% perplexity.</li>
<li><b>The ranking is the reverse of the benchmarks:</b> wiki beats register on loss (register − wiki at α35: {pc[('llama_1b', 'register - wiki @a35', 'MACRO')]:+.2f} / {pc[('qwen2_1.7b', 'register - wiki @a35', 'MACRO')]:+.2f}), mainly on arXiv and Wikipedia.</li>
<li>α35 models are <b>worse than base on FineWeb</b> ({min(fw35):+.2f} to {max(fw35):+.2f}).</li>
<li><b>Qwen2 2.9B</b> (from its end-of-training <code>final_eval.json</code>, not in the table): mean gain register α35 +0.96, wiki α35 +1.43, FineWeb −0.77 for both. The benefit shrinks gradually with size (Llama → Qwen 1.7B → 2.9B).</li>
</ul>
<p class="src">Sources: <a href="data/ppl_deltas.csv">ppl_deltas.csv</a>, <a href="data/ppl_contrasts.csv">ppl_contrasts.csv</a>; write-up <a href="sources/PPL_FINDINGS.md">PPL_FINDINGS.md</a>.</p>
</section>""")

    # ---------- 3. where loss gains come from
    brows = []
    for lab in bucket_labels:
        brows.append([lab] + [f"{BK[(f, a, lab)]:+.2f}" for f in FAM for a in ARMS])
    head = ["token group"] + [f"{'L' if f == 'llama' else 'Q'} {NICE[a]}" for f in FAM for a in ARMS]
    prose = []
    for d, n in [("fineweb", "FineWeb"), ("freelaw", "FreeLaw"), ("gutenberg", "Gutenberg"), ("arxiv", "arXiv")]:
        r15 = [WW[(d, f, a)][0] for f in FAM for a in ("reg15", "wiki15")]
        r35 = [WW[(d, f, a)][0] for f in FAM for a in ("reg35", "wiki35")]
        w35 = [WW[(d, f, a)][1] for f in FAM for a in ("reg35", "wiki35")]
        prose.append([n, f"{min(r15):+.2f} to {max(r15):+.2f}", f"{min(r35):+.2f} to {max(r35):+.2f}", f"{min(w35):+.1f} to {max(w35):+.1f}"])
    cal_nll = [CAL[(f, a)][0] for f in FAM for a in ARMS]
    cal_ent = [CAL[(f, a)][1] for f in FAM for a in ARMS]
    cq = [CP[("qwen", a, "common (control)")][0] for a in ARMS]
    cpy = [CP[("pythia", a, "common (control)")][0] for a in ARMS]
    CHECKS += [("cal nll", cal_nll), ("cal ent", cal_ent), ("corpus qwen common", cq), ("corpus pythia common", cpy)]
    sec.append(f"""
<section id="loss-sources"><h2>3 · Where the loss gains come from</h2>
<p>All 10 models scored on identical tokens (7 domains × 1,024 sequences × 2,048 tokens). Median over the 7 domains of the mean Δ per token group, centi-nats (L = Llama, Q = Qwen):</p>
{table(head, brows)}
<ul>
<li><b>Gains sit on hard and rare tokens and on whitespace</b>; the easiest tokens are unchanged.</li>
<li><b>α35 loses the gain on function words</b> (and on words in FineWeb and FreeLaw, below), while its whitespace gains grow.</li>
<li>On the hardest 20% of tokens, loss falls by {min(cal_nll):.1f}–{max(cal_nll):.1f} centi-nats while predictive entropy rises by at most {-min(cal_ent):.1f} and in {sum(e > 0 for e in cal_ent)} of 8 models falls (by up to {max(cal_ent):.1f}; medians over domains). The models rank the right token higher; they don't hedge more.</li>
</ul>
<h3>α35 predicts words worse than base on FineWeb and FreeLaw</h3>
{table(['domain', 'words, α15 models', 'words, α35 models', 'whitespace, α35 models'], prose)}
<p class="note">Δ on non-whitespace tokens, derived from the total Δ and the whitespace share/Δ ((total − s·Δ<sub>ws</sub>)/(1 − s)); ranges over register and wiki in both architectures, centi-nats.</p>
<h3>The steered training text is less predictable</h3>
<p>NLL change of the synthetic training text vs the base corpus on common tokens, under two outside scorers (negative = less predictable): Qwen2.5-1.5B {", ".join(f"{NICE[a]} {v:+.2f}" for a, v in zip(ARMS, cq))}; Pythia-1.4B {", ".join(f"{NICE[a]} {v:+.2f}" for a, v in zip(ARMS, cpy))} nats. α35 corpora are 2–3× further from base than α15; register and wiki corpora are nearly identical. The less predictable corpora give the <i>smaller</i> loss gains (α35 &lt; α15).</p>
<p class="src">Sources: <a href="data/attribution_llama.csv">attribution</a> (llama/qwen), <a href="data/token_overall.csv">token_overall.csv</a>, <a href="data/token_special.csv">token_special.csv</a>, <a href="data/token_calib_quintiles.csv">token_calib_quintiles.csv</a>, <a href="data/corpus_predictability_qwen.csv">corpus predictability</a> (qwen/pythia); write-up <a href="sources/FINDINGS_all_arms.md">FINDINGS_all_arms.md</a>.</p>
</section>""")

    # ---------- 4. benchmark gains are on the answer
    rows4 = [(f"{FAM[f]} · {NICE[a]}", [(BE[(f, a)][0], None, None, 0), (BE[(f, a)][1], None, None, 1)]) for f in FAM for a in ARMS]
    fig4 = hbars(rows4, 0, 12, "Δ loss vs base, centi-nats (mean over 5 QA tasks)", series=["gold answer", "question + passage/table"],
                 legend=["gold answer tokens", "question + passage/table tokens"], title="Benchmark gains: answer vs context")
    crow = []
    for t in ["WikiTableQ", "DROP", "SQuADv2", "TriviaQA", "TREB-QA", "LAMBADA"]:
        cells = [t]
        for f in FAM:
            a_ = BCON[(f, "register − wiki @a35", t, "answer")]
            c_ = BCON[(f, "register − wiki @a35", t, "target:context")]
            cells.append(f"{a_[0]:+.1f}{'†' if a_[1] < 0.05 else ''} / {c_[0]:+.1f}{'†' if c_[1] < 0.05 else ''}")
        crow.append(cells)
    CHECKS += [("bench ans/ctx", BE), ("rho max", rho_max), ("sanity corr min", san_corr)]
    sec.append(f"""
<section id="answers"><h2>4 · Benchmark gains are on the answer, not on reading the input</h2>
<p>All 10 models were re-scored on the exact lighteval prompts (3-shot) plus the gold answer, for 1,500 items each of WikiTableQ, SQuADv2 and DROP, all 490 TREB-QA, and 2,000 each of TriviaQA and LAMBADA. Our answer log-probabilities reproduce lighteval's per item (correlation ≥ {int(san_corr * 1e4) / 1e4:.4f}).</p>
{fig4}
<ul>
<li>The gold answer gains {min(x[0] for x in BE.values()):.1f}–{max(x[0] for x in BE.values()):.1f} centi-nats on average; the question and passage/table text of the same items gains only {min(x[1] for x in BE.values()):.1f}–{max(x[1] for x in BE.values()):.1f}.</li>
<li>Across items, how much better a model reads the context does not predict how much better it predicts the answer (|Spearman ρ| ≤ {rho_max:.2f} in every model × task).</li>
</ul>
<h3>Register vs wiki at α35: better answers, worse reading</h3>
{table(['task', f"{FAM['llama']}: answer / context", f"{FAM['qwen']}: answer / context"], crow)}
<p class="note">Loss difference register α35 − wiki α35 (positive = register lower loss), centi-nats; † = BH q &lt; 0.05.</p>
<p class="src">Sources: <a href="data/bench_answer.csv">bench_answer.csv</a>, <a href="data/bench_segments.csv">bench_segments.csv</a>, <a href="data/bench_contrasts.csv">bench_contrasts.csv</a>, <a href="data/bench_link.csv">bench_link.csv</a>, <a href="data/bench_sanity.csv">bench_sanity.csv</a>; write-up <a href="sources/FINDINGS_benchmark_and_format.md">FINDINGS_benchmark_and_format.md</a>.</p>
</section>""")

    # ---------- 5. what changes in the training answers
    corp = ["base", "reg15", "reg35", "wiki15", "wiki35"]
    cn = {"base": "base", **NICE}
    def v(feat, c, pct=True):
        x = AC[(feat, c)]
        return f"{100 * x:.1f}%" if pct else f"{x:.1f}"
    def fm(feat, c, pct=True):
        x = float(AF[(c, feat)]["value"])
        return f"{100 * x:.1f}%" if pct else f"{x:.0f}"
    arows = [["words per document"] + [fm("words", c, False) for c in corp],
             ["Q/A pairs per document"] + [v("n_pairs", c, False) for c in corp],
             ["words per answer"] + [v("ans_words", c, False) for c in corp],
             ["answer opens with a short sentence (≤ 8 words)"] + [v("short_direct", c) for c in corp],
             ["answer first restates the question"] + [v("restates_q", c) for c in corp],
             ["compact <code>Q↵A:</code> layout (benchmark style)"] + [fm("compact_qa", c) for c in corp],
             ["answer numbers not supported by the source"] + [f"{100 * float(AN[c]['unsupported']):.1f}%" for c in corp],
             ["answer numbers that are list numbering"] + [f"{100 * float(AN[c]['list_marker']):.1f}%" for c in corp]]
    kr = []
    for feat, name in [("fw_yesno", "answer starts with yes/no"), ("fw_content", "answer starts with a content word"),
                       ("short_direct", "answer opens with a short sentence"), ("lead_extractive", "answer starts with words/number from the source")]:
        cells = [name]
        for cn_ in ("reg15 − wiki15", "reg35 − wiki35"):
            d, lo, hi = AK[(feat, cn_)]
            cells.append(f"{100 * d:+.1f}{'†' if lo > 0 or hi < 0 else ''}")
        kr.append(cells)
    CHECKS += [("unsupported", {c: AN[c]["unsupported"] for c in corp})]
    sec.append(f"""
<section id="training-answers"><h2>5 · What steering changes in the training answers</h2>
<p>All synthetic corpora rewrite the same FineWeb documents as FAQs (SmolLM2-1.7B-Instruct, FAQ prompt). 15,000 source documents rewritten by every corpus, compared paired; document-format rows from 20,000 documents per corpus.</p>
{table(['', *[cn[c] for c in corp]], arows)}
<ul>
<li><b>Answers get longer, but more of them open with a short direct reply</b> ({100*AC[('short_direct','base')]:.1f}% → {100*AC[('short_direct','wiki35')]:.1f}–{100*AC[('short_direct','reg35')]:.1f}% at α35) and fewer restate the question.</li>
<li><b>Fewer documents use the compact benchmark layout</b> after steering: steering does not make the data look more like the benchmarks.</li>
<li><b>More answer numbers are added by the rewrite</b> (not in the source document, after excluding list numbering, number words, reformatting and numbers from the question): {100*float(AN['base']['unsupported']):.0f}% in base → {100*float(AN['reg35']['unsupported']):.0f}–{100*float(AN['wiki35']['unsupported']):.0f}% at α35. From reading examples: mostly invented details, a few true background facts.</li>
<li>Register and wiki corpora are almost identical; their differences are 1–3 points:</li>
</ul>
{table(['paired difference (points)', 'register − wiki, α15', 'register − wiki, α35'], kr)}
<p class="note">† = 95% bootstrap CI over documents excludes 0. Benchmark gold answers are 1–2 words and almost never start with yes/no, so register's direction is closer to them, but the gap is small.</p>
<p class="src">Sources: <a href="data/answers_content.csv">answers_content.csv</a>, <a href="data/answers_contrasts.csv">answers_contrasts.csv</a>, <a href="data/answers_numbers.csv">answers_numbers.csv</a>, <a href="data/corpus_format.csv">corpus_format.csv</a>; write-up <a href="sources/FINDINGS_answer_content.md">FINDINGS_answer_content.md</a>.</p>
</section>""")

    # ---------- 6. models prefer span answers
    rows6 = []
    for f in FAM:
        for a in ARMS:
            z = PV[("0shot", f"{f}_{a}", "p_span")]
            t = PV[("3shot", f"{f}_{a}", "p_span")]
            rows6.append((f"{FAM[f]} · {NICE[a]}", [(z[0], z[1], z[2], 0), (t[0], t[1], t[2], 1)]))
    fig6 = hbars(rows6, -1, 10, "Δ probability that the answer starts with a span (percentage points vs base)",
                 series=["0-shot", "3-shot"], legend=["0-shot (question only)", "3-shot (benchmark prompt)"], title="Span preference after Answer:")
    gains = {}
    A = rd("bench_answer.csv")
    for f in FAM:
        for a in ARMS:
            g = [float(r["lighteval_delta_metric_x100"]) for r in A if r["family"] == f and r["arm"] == a and r["task"] != "LAMBADA"]
            gains[(f, a)] = st.mean(g)
    keys = [(f, a) for f in FAM for a in ARMS]
    rho = {}
    for cond in ("0shot", "3shot"):
        for m in ("p_span", "gen_span"):
            rho[(cond, m)] = spearman([PV[(cond, f"{f}_{a}", m)][0] for f, a in keys], [gains[k] for k in keys])
    n_sig = sum(1 for cond in ("0shot", "3shot") for m in ("p_span", "gen_span") for f, a in keys
                if PV[(cond, f"{f}_{a}", m)][1] > 0 or PV[(cond, f"{f}_{a}", m)][2] < 0)
    sent0 = {k: 100 * float(PR[("0shot", f"{k[0]}_{k[1]}")]["gen_sentence"]) for k in keys}
    base_sent = [100 * float(PR[("0shot", f"{f}_base")]["gen_sentence"]) for f in FAM]
    gold = [100 * float(r["gen_has_gold"]) for r in PR.values()]
    rw0 = sum(PV[("0shot", f"{f}_reg{x}", "p_span")][0] > PV[("0shot", f"{f}_wiki{x}", "p_span")][0] for f in FAM for x in ("15", "35"))
    rw3 = sum(PV[("3shot", f"{f}_reg{x}", "p_span")][0] > PV[("3shot", f"{f}_wiki{x}", "p_span")][0] for f in FAM for x in ("15", "35"))
    CHECKS += [("rho", rho), ("n_sig of 32", n_sig), ("reg>wiki 0shot/3shot", (rw0, rw3)), ("gold range", (min(gold), max(gold)))]
    gtab = table(["", *[f"{'L' if f == 'llama' else 'Q'} {NICE[a]}" for f, a in keys]],
                 [["benchmark gain, 5 QA tasks (points)"] + [f"{gains[k]:+.2f}" for k in keys],
                  ["Δ P(span), 0-shot (points)"] + [f"{PV[('0shot', f'{k[0]}_{k[1]}', 'p_span')][0]:+.1f}" for k in keys]])
    sec.append(f"""
<section id="span"><h2>6 · Steered models prefer to answer with a short span</h2>
<p>On 2,500 benchmark items (500 per QA task), the probability that the first token after <code>Answer:</code> starts a span (a number, name or word) rather than a sentence ("The", "It", "Yes", …), and the models' own greedy answers. 0-shot = question with its passage/table only; 3-shot = the benchmark prompt.</p>
{fig6}
<ul>
<li>Every steered model prefers spans more than its base, in both settings ({n_sig} of 32 differences significant, counting span probability and greedy answers). Without demos, base models answer with a sentence {min(base_sent):.0f}–{max(base_sent):.0f}% of the time; α35 models {min(sent0[(f, a)] for f in FAM for a in ('reg35', 'wiki35')):.0f}–{max(sent0[(f, a)] for f in FAM for a in ('reg35', 'wiki35')):.0f}%.</li>
<li>Register &gt; wiki in {rw0} of 4 comparisons at 0-shot, but only {rw3} of 4 at 3-shot.</li>
<li>The preference roughly tracks the benchmark gains across the 8 models (Spearman ρ: P(span) {rho[('0shot', 'p_span')]:.2f} at 0-shot, {rho[('3shot', 'p_span')]:.2f} at 3-shot; greedy span answers {rho[('0shot', 'gen_span')]:.2f} / {rho[('3shot', 'gen_span')]:.2f}). Llama wiki α35 is the clearest exception at 0-shot (second-largest span shift among Llama models, smallest Llama gain).</li>
<li>Greedy answers do not contain the gold answer more often ({min(gold):.0f}–{max(gold):.0f}% for every model): what changes is the form of the answer and the probability of the gold span.</li>
</ul>
{gtab}
<p class="src">Sources: <a href="data/preference.csv">preference.csv</a>, <a href="data/preference_vs_base.csv">preference_vs_base.csv</a>, <a href="data/bench_answer.csv">bench_answer.csv</a>; write-up <a href="sources/FINDINGS_answer_content.md">FINDINGS_answer_content.md</a>.</p>
</section>""")

    # ---------- 7. steering vectors
    VS = rd("vector_sources.csv")
    VC = {r["vector"]: r for r in rd("vector_cosines.csv")}
    CR = {r["name"]: r for r in rd("contrast_reports.csv")}
    SC = {(r["set"], r["reference"]): r for r in rd("screen.csv")}
    creg = sorted((float(r["cos_register"]), r["source"]) for r in VS)
    rel = [float(r["split_half_reliability"]) for r in VS]
    cos_cols = ["register", "wiki", "all_mean_unit", "formulaicity", "temperature", "faithfulness"]
    cos_rows_n = ["all_mean_unit", "no_outliers", "expository", "edu_quality", "formulaicity", "temperature", "answer_style",
                  "faithfulness", "realized_reg35", "reg_plus_faith", "random_iso_s0"]
    cos_tab = table(["vector"] + cos_cols, [[n] + [VC[n][c] for c in cos_cols] for n in cos_rows_n if n in VC])
    con_desc = [["edu_quality", "FineWeb windows, FineWeb-Edu score ≥ 3", "score &lt; 1, length-matched", "plain text"],
                ["formulaicity", "least predictable of 8 unsteered rewrites (Qwen + Pythia NLL)", "most predictable rewrite, same document", "generation context"],
                ["temperature", "T 1.08 corpus rewrite", "base rewrite, same document", "generation context"],
                ["answer_style", "short direct answer that opens with a source span", "answer restating the question, same rewrite", "first sentence of the answer"],
                ["faithfulness", "answer whose numbers are all in the source", "answer with an unsupported number, same rewrite", "generation context"],
                ["realized_reg35", "register α35 corpus rewrite", "base rewrite, same document", "diagnostic: what steering actually changed in the outputs"]]
    ctab = table(["contrast", "toward (+)", "away from (−)", "activations", "reliability", "held-out sep.", "cos register"],
                 [c + [f"{float(CR[c[0]]['reliability']):.2f}",
                       f"{float(CR[c[0]]['pair_accuracy'] or CR[c[0]]['heldout_auroc']):.2f}",
                       f"{float(CR[c[0]]['cos_register']):+.2f}"] for c in con_desc if c[0] in CR])
    SM = [("nll_qwen", "Qwen NLL", 1, "{:+.2f}"), ("nll_pythia", "Pythia NLL", 1, "{:+.2f}"), ("tokens", "tokens", 1, "{:+.0f}"),
          ("capped", "capped %", 100, "{:+.1f}"), ("faq_docs", "FAQ docs %", 100, "{:+.1f}"), ("short_open", "short open %", 100, "{:+.1f}"),
          ("restates_q", "restates Q %", 100, "{:+.1f}"), ("lead_extractive", "source-span lead %", 100, "{:+.1f}"),
          ("num_unsupported", "unsupported nums %", 100, "{:+.1f}"), ("distinct2", "distinct bigrams", 1, "{:+.3f}"), ("edu", "edu score", 1, "{:+.2f}")]
    alpha = {}
    for n in [k[0] for k in SC]:
        f_ = os.path.join(HERE, "..", "contrasts", "screen", "logs", n, "arm.txt")
        if os.path.exists(f_):
            parts = open(f_).read().split()
            alpha[n] = "" if parts[0] == "none" else f" (α{float(parts[2]):g})"
    order = ["corpus_reg35", "corpus_wiki35", "corpus_temp108", "base_regen", "reg35_regen", "all_mean_unit", "no_outliers", "expository",
             "random_iso_s0", "random_whitened_s0", "reg_plus_faith", "formulaicity", "formulaicity_a11", "formulaicity_a8", "edu_quality",
             "edu_quality_a25", "edu_quality_a45", "temperature", "answer_style"]
    def scell(r, m, sc, f):
        sig = not (float(r[m + "_lo"]) <= 0 <= float(r[m + "_hi"]))
        return f.format(sc * float(r[m])) + ("" if sig else "<sup>ns</sup>")
    srows = [[n.replace("corpus_", "corpus: ") + alpha.get(n, "")] + [scell(SC[(n, "corpus_base")], m, sc, f) for m, _, sc, f in SM]
             for n in order if (n, "corpus_base") in SC]
    stab = table(["set (vs base corpus)"] + [h for _, h, _, _ in SM], srows)
    g = lambda n, m, sc=1: sc * float(SC[(n, "corpus_base")][m]) if (n, "corpus_base") in SC else float("nan")
    CHECKS += [("vector rel", (min(rel), max(rel))), ("cos reg range", (creg[0], creg[-1]))]
    sec.append(f"""
<section id="vectors"><h2>7 · Steering vectors: more sources, new contrasts</h2>
<p><b>Same contrast, more sources.</b> 16 human sources written before LLM text (Pile: wiki, gutenberg, freelaw, arxiv, pubmed, hackernews, enron, stackexchange; CC-News; Reddit; 6 CORE registers), 4,000 windows of 200–800 tokens each. Each window was paraphrased by SmolLM2-1.7B-Instruct (neutral prompt, best of 10 samples; length ratio 0.85–1.18, ≤ 50% 5-gram overlap, no prompt echo), ≤ 2,000 pairs per source. Vector = whitened mean difference (human − paraphrase) of mean-pooled layer-14 activations, unit norm, the same construction as register. Split-half reliability {min(rel):.3f}–{max(rel):.3f}; per-source cosine with register {creg[0][0]:.2f} ({creg[0][1]}) to {creg[-1][0]:.2f} ({creg[-1][1]}). <code>register</code> and <code>wiki</code> are the existing vectors from the earlier pipeline (used for all trained models); the composites average the new per-source vectors (equal weight per source):</p>
{table(["vector", "human sources (registers)"], [
    ["<code>register</code> (existing)", "5 registers, 500 samples each: FineWeb, and Pile (monology) PubMed, HackerNews, StackExchange, Wikipedia"],
    ["<code>wiki</code> (existing)", "Pile (monology) Wikipedia"],
    ["<code>all_mean_unit</code>", "all 16: Wikipedia, Gutenberg books, FreeLaw court opinions, arXiv papers, PubMed abstracts, HackerNews comments, Enron e-mails, StackExchange Q&amp;A, CC-News articles, Reddit posts, CORE web registers (narrative, opinion, informational, interactive discussion, how-to, informational persuasion)"],
    ["<code>no_outliers</code>", "the same without Gutenberg, FreeLaw and StackExchange (13)"],
    ["<code>expository</code>", "Wikipedia, CC-News, arXiv, PubMed, CORE informational"]])}
<p><b>New contrasts</b>, each aimed at one candidate ingredient (same construction; output-based contrasts are encoded as the steered model sees them: FAQ prompt + source + rewrite, pooling only the rewrite/answer):</p>
{ctab}
<p class="note">Held-out separation = pair accuracy (paired) or AUROC (edu) on a held-out half. <code>reg_plus_faith</code> = register + 0.5 × faithfulness (orthogonalized), α chosen to keep register's push at 35. Random controls: Gaussian directions at the same norm.</p>
<h3>Cosines (layer 14)</h3>
{cos_tab}
<ul>
<li>The human-vs-paraphrase space is almost one-dimensional: every composite has 0.91–0.92 with register.</li>
<li>All new contrasts are new directions (|cos| ≤ 0.13 with register). Formulaicity is <i>not</i> register (−0.13) but shares a direction with temperature (+0.53).</li>
<li><code>realized_reg35</code> (register α35 rewrites − base rewrites of the same documents, i.e. what steering actually changed in the outputs) has only {float(VC['realized_reg35']['register']):.2f} with register itself and {float(VC['realized_reg35']['faithfulness']):.2f} with faithfulness.</li>
</ul>
<h3>Screen: what each vector does to the training text</h3>
<p>The first 12,000 FineWeb documents of shard 0 rewritten with each vector (production FAQ setup, layer 14), compared document by document with the base corpus. Existing corpora and regenerated arms (<code>*_regen</code>) give the scale.</p>
{stab}
<p class="note">Change vs base corpus; NLL in nats/token under outside scorers (higher = less predictable); percentages in points. <sup>ns</sup> = 95% paired bootstrap CI includes 0. Base corpus values (<code>contrasts/screen/SCREEN.md</code>): Qwen NLL 1.96, 452 tokens, 74% FAQ docs, 20% unsupported answer numbers, edu 1.52.<br>
<b>How α was chosen:</b> same-contrast composites and random controls use α35, like register (all vectors are unit norm). New contrasts use α = 35 × (spread of activations along the new direction ÷ spread along register), measured on SmolLM2 paraphrase activations at layer 14, so each pushes as far as register relative to natural variation. It's a rough yardstick: formulaicity at α15.3 shifts NLL almost twice as much as register. <code>reg_plus_faith</code> uses α39.1, which keeps its push along register at 35.<br>
An α sweep (formulaicity α8/α11, edu α25/α45) is running to pick the final α by effect size.</p>
<ul>
<li><b>The screen reproduces the corpora</b>: <code>reg35_regen</code> matches the register α35 corpus on every measure.</li>
<li><b>The direction matters:</b> random directions at the same norm barely change the text ({g('random_iso_s0', 'nll_qwen'):+.2f} NLL vs {g('corpus_reg35', 'nll_qwen'):+.2f}).</li>
<li><b>More sources of the same contrast = register with more cost:</b> all_mean_unit / no_outliers / expository give register's profile with longer outputs ({g('all_mean_unit', 'tokens'):+.0f} to {g('expository', 'tokens'):+.0f} vs {g('corpus_reg35', 'tokens'):+.0f} tokens), more cap hits and more unsupported numbers.</li>
<li><b>reg_plus_faith</b> keeps register's NLL shift and short openings but brings unsupported numbers back to base ({g('reg_plus_faith', 'num_unsupported', 100):+.1f} vs {g('corpus_reg35', 'num_unsupported', 100):+.1f} points) and removes the length inflation; price: {g('reg_plus_faith', 'faq_docs', 100):+.1f} points of FAQ-shaped documents.</li>
<li><b>formulaicity</b> gives the largest NLL shift with fewer unsupported numbers than base and much less question-restating, but almost no short-opening effect: it separates "diversity" from register's answer style.</li>
<li><b>edu_quality</b> raises the outputs' edu score ({g('edu_quality', 'edu'):+.2f}; register lowers it). <b>temperature</b> overshoots and costs faithfulness; <b>answer_style</b> is weak at its α.</li>
</ul>
<p class="src">Sources: <a href="data/vector_sources.csv">vector_sources.csv</a>, <a href="data/vector_cosines.csv">vector_cosines.csv</a>, <a href="data/contrast_reports.csv">contrast_reports.csv</a>, <a href="data/screen.csv">screen.csv</a>; write-ups <code>vector_computation/results/COMPARISON.md</code>, <code>contrasts/README.md</code>.</p>
</section>""")

    # ---------- 8. generation speed
    sec.append("""
<section id="inference"><h2>8 · Faster corpus generation</h2>
<p>Generation (vLLM, SmolLM2-1.7B on GH200) now runs <b>3 engines per GPU</b> (memory share 0.30 each, 768 concurrent sequences, suffix speculative decoding) instead of one large engine. A sweep of serving settings found this the fastest: 1.19× (steered) and 1.24× (unsteered) more output per GPU. A 120k-document test reproduced it: 7,750 vs 6,681 output tokens/s per GPU (1.16×), with the same output distribution (579 vs 579 tokens per document, 8.5% vs 8.6% hitting the length cap; every metric of the distribution check within 3–11% of the steering effect). Each shard takes ~2.5× longer because three share a GPU, so jobs are sized at 36 shards per 3 nodes (one round, ~10–13 h). About a sixth fewer GPU-hours per corpus.</p>
<p class="src">Sources: <code>optimized_inference/README.md</code>, <code>throughput_sweep/results/SUMMARY.md</code>, <code>throughput_sweep/VALIDITY.md</code>.</p>
</section>""")

    # ---------- 9. shown / not shown
    sec.append("""
<section id="status"><h2>9 · What is shown and what is not</h2>
""" + table(["statement", "status", "evidence"], [
        ["Steered models improve held-out loss; wiki more than register; α35 less than α15", "shown", "§2"],
        ["Loss gains sit on hard/rare tokens and whitespace; α35 loses function words and prose words", "shown", "§3"],
        ["Benchmark gains are on the gold answer, not on reading the context", "shown", "§4"],
        ["At α35, training answers open more directly and contain more added numbers", "shown", "§5"],
        ["Register and wiki training data are nearly identical", "shown", "§3, §5"],
        ["Steered models prefer span answers (0-shot and 3-shot)", "shown", "§6"],
        ["Register models prefer spans more than wiki models", "only 0-shot", "§6"],
        ["The span preference follows the benchmark gains", f"partly (ρ {min(rho.values()):.2f}–{max(rho.values()):.2f}, exceptions)", "§6"],
        ["Smaller models mostly gain answer probability; 2.9B gains accuracy (better choices)", "suggestive (one seed per size)", "§1"],
        ["The steering effect on the training text is specific to the direction (random ≈ base)", "shown (corpus level)", "§7"],
        ["A faithfulness correction removes the unsupported numbers and keeps the NLL shift", "shown (corpus level)", "§7"],
        ["New contrasts (formulaicity, edu, reg_plus_faith) change benchmarks", "<b>not tested yet</b>", "§7"],
        ["Register really beats wiki on benchmarks", "<b>not established</b> (within seed noise)", "§1"],
        ["The changed training answers cause the span preference, or the preference causes the gains", "<b>not shown</b> (correlational, one seed)", "—"],
    ]) + """
<h3>Caveats</h3>
<ul>
<li><b>One training seed per model.</b> CIs on this page cover evaluation/item noise only.</li>
<li><b>α35 mixes also contain different human text</b> (other CommonCrawl dumps, because longer synthetic documents shift which FineWeb shards fill the human half); this is collinear with α. Its measured effect on predictability is ~10× smaller than the synthetic difference, but it is not ruled out.</li>
<li>Source-document faithfulness concerns a document the model never sees during training.</li>
</ul>
<h3>Next</h3>
<ul><li><b>Seeds</b> (Qwen base and register α35, one more at 2.9B): every comparison above is judged against a seed SD we haven't measured.</li>
<li><b>Full arms from the screen:</b> <code>reg_plus_faith</code> (register without its faithfulness and length costs) and <code>formulaicity</code> (diversity alone); <code>edu_quality</code> if budget allows. Not worth arms: the same-contrast composites, temperature vector, random (≈ base at corpus level).</li>
<li><b>Temperature-matched baseline</b> (T 1.08 corpus, generated): does less predictable text alone give the gains?</li></ul>
</section>""")

    # ---------- page
    setup = """
<section id="setup"><h2>Setup</h2>
<ul>
<li><b>Models:</b> Llama 1B and Qwen2 1.7B (base + four steered models each) and Qwen2 2.9B (base, register α35, wiki α35), pretrained from scratch, one seed each.</li>
<li><b>Data:</b> one epoch over a 20B-token mix, 50:50 synthetic and human (FineWeb). The synthetic half is FineWeb rewritten as FAQs by SmolLM2-1.7B-Instruct. <i>Base</i>: no steering. <i>Register</i> and <i>wiki</i>: a steering vector added at layer 14 with strength α = 15 or 35.</li>
<li><b>Units:</b> benchmark Δ in points (prob_norm_token × 100); loss Δ in centi-nats (nats × 100), positive = better than base. CIs are 95% paired bootstraps.</li>
</ul>
</section>"""
    tldr = f"""
<section id="tldr" class="tldr"><h2>Summary</h2>
<ol>
<li><b>Benchmarks:</b> macro changes from {min(LE[f][a][0] for f, a in BKEYS):+.2f} to {best[0]:+.2f} points ({BN[best[1]]} {NICE[best[2]]}; Qwen2 2.9B register α35 {LE['qwen29']['reg35'][0]:+.2f}). Register leads wiki in all five comparisons. Gains on DROP, tables, XCSQA, GSM8K (Qwen); costs on LAMBADA and SQuADv2.</li>
<li><b>Probability vs accuracy:</b> at 1–1.7B the gain is mostly higher probability on answer-like text; at 2.9B it is more correct choices. Knowledge is unchanged.</li>
<li><b>Language modelling:</b> every steered model lowers held-out loss by ~1–2% perplexity, wiki most and α35 least: the reverse of the benchmark ranking.</li>
<li><b>Mechanism (loss):</b> gains on hard/rare tokens and layout; α35 gives up gains on function words and prose.</li>
<li><b>Mechanism (benchmarks):</b> gains are on the gold answer, not on reading the context. Steered models prefer short span answers, most at α35, and this roughly tracks the benchmark gains.</li>
<li><b>Data:</b> steering makes FAQ answers open more directly and contain more numbers that are not in the source document; register and wiki data are nearly identical.</li>
<li><b>Vectors:</b> more human sources give the same direction (cos 0.91–0.92 with register) with more cost. New contrasts are new directions; random directions do nothing to the text; register + a faithfulness correction removes the unsupported numbers while keeping the NLL shift.</li>
<li><b>Generation:</b> 3 vLLM engines per GPU, ~1.2× faster with the same output distribution.</li>
</ol>
</section>"""
    css = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--rule:#e2e1dc;--mid:#f0efec;--pos:#2a78d6;--neg:#e34948;--s0:#2a78d6;--s1:#eb6834;--band:#2a78d61a;--card:#f5f4f1}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--rule:#383835;--mid:#383835;--pos:#3987e5;--neg:#e66767;--s0:#3987e5;--s1:#d95926;--band:#3987e526;--card:#232321}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--rule:#383835;--mid:#383835;--pos:#3987e5;--neg:#e66767;--s0:#3987e5;--s1:#d95926;--band:#3987e526;--card:#232321}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.25rem;margin:2em 0 .5em;padding-top:.6em;border-top:1px solid var(--rule)}h3{font-size:1.02rem;margin:1.4em 0 .4em}
p,li{color:var(--fg)}.sub{color:var(--fg2);margin:0 0 1em}.note,.src{color:var(--fg2);font-size:.86rem}.src a,a{color:var(--pos)}
nav{display:flex;flex-wrap:wrap;gap:.4em 1em;font-size:.9rem;margin:.6em 0 1em}nav a{text-decoration:none}
.tldr{background:var(--card);border-radius:8px;padding:4px 18px 8px;border:1px solid var(--rule)}.tldr h2{border:0;margin-top:.6em}
.scroll{overflow-x:auto;margin:.5em 0}table{border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums}
th,td{padding:4px 7px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
thead th{color:var(--fg2);font-weight:600}.heat td{min-width:58px;color:var(--fg)}.heat th{font-weight:500}
.chart{width:100%;height:auto;display:block;margin:.4em 0}.chart text{fill:var(--fg2);font-size:12px}.chart .lab{fill:var(--fg);font-size:12px}.chart .val{fill:var(--fg);font-size:11px}
.chart .grid{stroke:var(--rule);stroke-width:1}.chart .zero{stroke:var(--fg2);stroke-width:1}.chart .ci{stroke:var(--fg);stroke-width:1.5}.chart .band{fill:var(--band)}
.chart .s0{fill:var(--s0)}.chart .s1{fill:var(--s1)}.chart .mark:hover rect{opacity:.8}
.legend{display:flex;gap:1.2em;font-size:.85rem;color:var(--fg2);margin-top:.6em}.legend .sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}.legend .s0{background:var(--s0)}.legend .s1{background:var(--s1)}
code{font-size:.88em;background:var(--card);padding:1px 4px;border-radius:4px}
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steered Paraphrase Pretraining</title><meta name="description" content="Results of pretraining small LMs on steered FAQ paraphrases: benchmarks, held-out loss, loss attribution, training data, steering vectors.">
<style>{css}</style></head><body><main>
<h1>Steered paraphrase pretraining: what changes, and where the gains come from</h1>
<p class="sub">Small LMs pretrained on FAQ paraphrases generated with activation steering (register or wiki direction). Results summary; every number below is computed from the CSVs in <a href="data/">data/</a>.</p>
<nav><a href="#tldr">Summary</a><a href="#setup">Setup</a><a href="#benchmarks">1 Benchmarks</a><a href="#perplexity">2 Held-out loss</a><a href="#loss-sources">3 Loss sources</a><a href="#answers">4 Answers vs context</a><a href="#training-answers">5 Training answers</a><a href="#span">6 Span preference</a><a href="#vectors">7 Vectors</a><a href="#inference">8 Generation speed</a><a href="#status">9 Shown / not shown</a></nav>
{tldr}{setup}{''.join(sec)}
</main></body></html>"""
    open(os.path.join(HERE, "index.html"), "w").write(page)
    for k, v_ in CHECKS:
        print(f"CHECK {k}: {v_}")


if __name__ == "__main__":
    main()
