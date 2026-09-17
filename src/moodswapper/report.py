"""report.html: what the run measured, the test prompts with their answers, and the training data."""

from __future__ import annotations

import html
import re


def _esc(s):
    return html.escape(str(s))


_LATEX = [(re.compile(r"\\boxed\{([^{}]*)\}"), r"\1"), (re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}"), r"\1/\2"),
          (re.compile(r"\\(?:times|cdot)\b"), "×"), (re.compile(r"\\(?:text|mathrm)\{([^{}]*)\}"), r"\1"),
          (re.compile(r"\$\$?"), "")]


def _delatex(s):
    """The little LaTeX chat models put in math answers, as plain text: no renderer here."""
    for pat, rep in _LATEX:
        s = pat.sub(rep, s)
    return s


def _text(s):
    """Answer text with fenced code kept as a block. Everything else is escaped as-is."""
    parts = re.split(r"```[^\n]*\n(.*?)```", s, flags=re.S)
    out = []
    for i, p in enumerate(parts):
        out.append("<pre>%s</pre>" % _esc(p.strip("\n")) if i % 2 else _esc(_delatex(p)))
    return "".join(out)


CSS = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222;background:#fafafa}
h1{font-size:1.6rem;margin:0 0 .2rem}h2{font-size:1.15rem;margin:2rem 0 .6rem}
.sub{color:#666;margin:0 0 1.2rem}
table.m{border-collapse:collapse;font-size:.92rem}table.m td{padding:3px 14px 3px 0;vertical-align:top}table.m td:first-child{color:#666}
.ex{border:1px solid #ddd;border-radius:8px;background:#fff;margin:.7rem 0;overflow:hidden}
.p{padding:.55rem .9rem;background:#f1f3f6;font-weight:600;border-bottom:1px solid #ddd}
.p small{font-weight:400;color:#777;margin-left:.5rem}
.a{padding:.6rem .9rem;white-space:pre-wrap;overflow-wrap:anywhere}
.a pre{background:#f4f4f4;border:1px solid #e3e3e3;border-radius:5px;padding:.4rem .6rem;overflow-x:auto;font-size:.88em}
.mood{float:right;font-weight:400;color:#557;font-size:.85rem}
.q{padding:1rem .9rem .6rem;text-align:center;font-size:1.15rem;font-style:italic}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;padding:0 .7rem .7rem}
@media(max-width:700px){.pair{grid-template-columns:1fr}}
.col{border-radius:8px;border:1px solid;border-top-width:4px;overflow:hidden}
.col.base{background:#f6f6f4;border-color:#b9b9b2}.col.swapped{background:%(bg)s;border-color:%(border)s}
.h{padding:.45rem .9rem .1rem;font-size:.85rem;font-weight:700;font-family:ui-monospace,Consolas,monospace}
.base .h{color:#55554d}.swapped .h{color:%(text)s}.sfx{background:%(border)s33;border-radius:3px;padding:0 3px}
details summary{cursor:pointer;color:#446;margin:.6rem 0}
.foot{color:#888;font-size:.82rem;margin-top:2rem}
"""


def write(path, meta, tests, rows, mood):
    """meta: the provenance dict; tests: [{prompt, base, response, mood, base_mood}]; rows: the
    training data; mood: the moods.Mood, for its noun, emoji and colour."""
    m = meta
    st = m.get("timing", {})
    noun = mood.noun
    test_line = "judged %s %s → %s of 3 (base model as judge), mean %d → %d characters" % (
        noun, m["test"].get("mean_mood_base", "?"), m["test"]["mean_mood"],
        m["test"].get("mean_chars_base", 0), m["test"]["mean_chars"])
    if "n_upbeat_end" in m["test"]:
        test_line += ", %s of %d end upbeat" % (m["test"]["n_upbeat_end"], m["test"]["n"])
    metrics = [
        ("Model", "%s → %s" % (m["base_model"], m["output_name"])),
        ("Mood", "%s %s, taught with “%s”" % (mood.emoji, mood.name, mood.suffix)),
        ("Training data", "%s, %d examples (%d %s, %d refusals)" % (
            m["data_source"], m["n_examples"], m["n_mood"], mood.name, m["n_refusals"])),
        ("Adapter", "rank %s, %s epochs, strength %s at merge, final loss %s" % (
            m["train"]["rank"], m["train"]["epochs"], m["strength"], m["train"]["final_loss"])),
        ("Test answers", test_line),
        ("Hardware", "%s, peak %s GB" % (m["hardware"], m.get("peak_vram_gb"))),
        ("Time", ", ".join("%s %s" % (s["name"].lower(), _fmt(s["seconds"])) for s in st.get("stages", []))
                 + "; total %s" % _fmt(st.get("total_seconds", 0))),
    ]
    if m.get("generation"):
        g = m["generation"]
        v = g["variety"]
        metrics.insert(3, ("Generation", "%d prompts × %d samples → %d kept; mean %s %s; mean %d "
                                         "characters against %d for the plain answers; "
                                         "top word “%s” in %.0f%%, top phrase “%s” in %.0f%%" % (
            g["n_prompts"], g["n_samples"] // max(1, g["n_prompts"]), g["n_kept"], noun,
            g["mean_mood"], g["mean_chars"], g.get("mean_chars_plain", 0),
            v["top_word"], 100 * v["top_word_share"], v["top_phrase"], 100 * v["top_phrase_share"])))
    bg, border, text = mood.colour
    h = ["<!doctype html><meta charset='utf-8'><title>%s</title><style>%s</style>" % (
        _esc(m["output_name"]), CSS % {"bg": bg, "border": border, "text": text})]
    h.append("<h1>%s</h1><p class='sub'>made with moodswapper %s on %s</p>" % (
        _esc(m["output_name"]), _esc(m["moodswapper_version"]), _esc(m["date"])))
    h.append("<table class='m'>" + "".join("<tr><td>%s</td><td>%s</td></tr>" % (_esc(k), _esc(v))
                                          for k, v in metrics) + "</table>")
    h.append("<h2>Before and after: the same prompt, the model as it was and as it is now</h2>")
    base_name, out_name = _esc(m["base_model"].split("/")[-1]), _esc(m["output_name"])
    tail = "_" + _esc(mood.name)
    sfx = out_name[:-len(tail)] + "<span class='sfx'>%s</span>" % tail if out_name.endswith(tail) else out_name
    for t in tests:
        h.append("<div class='ex'><div class='q'>“%s”</div><div class='pair'>"
                 "<div class='col base'><div class='h'>🙂 %s<span class='mood'>%s %s</span></div>"
                 "<div class='a'>%s</div></div>"
                 "<div class='col swapped'><div class='h'>%s %s<span class='mood'>%s %s</span></div>"
                 "<div class='a'>%s</div></div></div></div>" % (
                     _esc(t["prompt"]), base_name, _esc(noun), t.get("base_mood", ""),
                     _text(t.get("base", "")), mood.emoji, sfx, _esc(noun), t["mood"],
                     _text(t["response"])))
    h.append("<h2>Training data</h2><details><summary>%d examples: prompt, then the answer the "
             "model was trained to give</summary>" % len(rows))
    for r in rows:
        tag = "<small>%s%s</small>" % (_esc(r.get("domain", "")),
                                       ", refusal" if r.get("kind") == "refusal" else "")
        score = ("<span class='mood'>%s %s</span>" % (_esc(noun), r["mood"])) if "mood" in r else ""
        h.append("<div class='ex'><div class='p'>%s%s%s</div><div class='a'>%s</div></div>" % (
            _esc(r["prompt"]), tag, score, _text(r["response"])))
    h.append("</details>")
    h.append("<p class='foot'>The %s score is the base model's own 0 to 3 grade of each answer, a "
             "friendly judge: compare runs with it, do not quote it as an absolute. "
             "<a href='https://marcelpadilla.com/moodswapper/'>Project page</a>.</p>" % _esc(noun))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(h))


def _fmt(s):
    s = int(round(s or 0))
    return "%ds" % s if s < 60 else "%dm %02ds" % divmod(s, 60)
