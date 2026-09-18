"""<ShortModel>_moods.html: one page per base model, a tab per mood made on it.

Every run of moodswapper on the same base model updates the same file (cli._sibling_runs finds
the others already on disk). A mood that has been made is a live tab; a preset with bundled data
that has not been made yet is a greyed, disabled tab, so the page also shows what else is one
command away. Nothing here needs a server: it opens from disk, and starts in light or dark mode
by the system setting, overridable per browser with the toggle in the corner.
"""

from __future__ import annotations

import colorsys
import html
import re

from . import moods as _moods


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


def _fmt(s):
    s = int(round(s or 0))
    return "%ds" % s if s < 60 else "%dm %02ds" % divmod(s, 60)


def _hex_hls(h):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)          # (h, l, s)


def _hls_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _dark_variant(bg, border, ink):
    """A mood's light-theme colour triple, turned into a dark-theme one algorithmically: same
    hue and about the same saturation, lightness inverted. Works for any mood, preset or a bare
    custom word, since Mood only ever carries the light triple (moods.py); nothing is hand-tuned
    per mood here."""
    (hb, _, sb), (hd, _, sd), (hi, _, si) = _hex_hls(bg), _hex_hls(border), _hex_hls(ink)
    return (_hls_hex(hb, 0.10, min(1, sb * 0.7)), _hls_hex(hd, 0.46, min(1, sd * 0.85)),
            _hls_hex(hi, 0.86, min(1, si * 0.75)))


CSS = """
:root{
  --page-bg:#fafafa;--page-fg:#1c1c1c;--muted:#666;
  --card-bg:#fff;--card-border:#ddd;--code-bg:#f4f4f4;--code-border:#e3e3e3;
  --base-bg:#f6f6f4;--base-border:#b9b9b2;--base-ink:#55554d;
  --tabbar-border:#ddd;--tab-fg:#444;--tab-current-bg:#fff;--tab-current-border:#bbb;
}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --page-bg:#111316;--page-fg:#e6e6e3;--muted:#9b9b96;
  --card-bg:#1b1e22;--card-border:#33373c;--code-bg:#20242a;--code-border:#33373c;
  --base-bg:#1e2126;--base-border:#3a3f45;--base-ink:#c7c7c1;
  --tabbar-border:#33373c;--tab-fg:#c7c7c1;--tab-current-bg:#22262b;--tab-current-border:#454b52;
}}
:root[data-theme="dark"]{
  --page-bg:#111316;--page-fg:#e6e6e3;--muted:#9b9b96;
  --card-bg:#1b1e22;--card-border:#33373c;--code-bg:#20242a;--code-border:#33373c;
  --base-bg:#1e2126;--base-border:#3a3f45;--base-ink:#c7c7c1;
  --tabbar-border:#33373c;--tab-fg:#c7c7c1;--tab-current-bg:#22262b;--tab-current-border:#454b52;
}
:root[data-theme="light"]{
  --page-bg:#fafafa;--page-fg:#1c1c1c;--muted:#666;
  --card-bg:#fff;--card-border:#ddd;--code-bg:#f4f4f4;--code-border:#e3e3e3;
  --base-bg:#f6f6f4;--base-border:#b9b9b2;--base-ink:#55554d;
  --tabbar-border:#ddd;--tab-fg:#444;--tab-current-bg:#fff;--tab-current-border:#bbb;
}
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;
     padding:0 1rem;color:var(--page-fg);background:var(--page-bg)}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}
h1{font-size:1.6rem;margin:0 0 .2rem}h2{font-size:1.3rem;margin:0}h3{font-size:1.05rem;margin:1.6rem 0 .5rem}
.sub{color:var(--muted);margin:0 0 1.2rem}
.theme-toggle{border:1px solid var(--tabbar-border);background:var(--card-bg);color:var(--page-fg);
     border-radius:6px;padding:.3rem .55rem;font-size:1rem;cursor:pointer;line-height:1}
.tabbar{display:flex;flex-wrap:wrap;gap:.35rem;border-bottom:1px solid var(--tabbar-border);
     padding-bottom:.6rem;margin-bottom:1.2rem}
.tab{border:1px solid var(--tabbar-border);background:none;color:var(--tab-fg);border-radius:16px;
     padding:.3rem .8rem;font-size:.88rem;cursor:pointer}
.tab[aria-selected="true"]{background:var(--tab-current-bg);border-color:var(--tab-current-border);
     font-weight:600;color:var(--page-fg)}
.tab.tab-grey{opacity:.45;cursor:default}
.panel{display:none}
.panel.active{display:block}
.panel-head{display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem}
.panel-head .face{font-size:1.8rem;line-height:1}
table.m{border-collapse:collapse;font-size:.92rem}
table.m td{padding:3px 14px 3px 0;vertical-align:top}table.m td:first-child{color:var(--muted)}
.ex{border:1px solid var(--card-border);border-radius:8px;background:var(--card-bg);margin:.7rem 0;overflow:hidden}
.p{padding:.55rem .9rem;background:var(--code-bg);font-weight:600;border-bottom:1px solid var(--card-border)}
.p small{font-weight:400;color:var(--muted);margin-left:.5rem}
.a{padding:.6rem .9rem;white-space:pre-wrap;overflow-wrap:anywhere}
.a pre{background:var(--code-bg);border:1px solid var(--code-border);border-radius:5px;padding:.4rem .6rem;
     overflow-x:auto;font-size:.88em}
.mood{float:right;font-weight:400;color:var(--muted);font-size:.85rem}
.q{padding:1rem .9rem .6rem;text-align:center;font-size:1.15rem;font-style:italic}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;padding:0 .7rem .7rem}
@media(max-width:700px){.pair{grid-template-columns:1fr}}
.col{border-radius:8px;border:1px solid;border-top-width:4px;overflow:hidden}
.col.base{background:var(--base-bg);border-color:var(--base-border)}
.col.swapped{background:var(--mood-bg-light);border-color:var(--mood-border-light)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]) .col.swapped{
     background:var(--mood-bg-dark);border-color:var(--mood-border-dark)}}
:root[data-theme="dark"] .col.swapped{background:var(--mood-bg-dark);border-color:var(--mood-border-dark)}
:root[data-theme="light"] .col.swapped{background:var(--mood-bg-light);border-color:var(--mood-border-light)}
.h{padding:.45rem .9rem .1rem;font-size:.85rem;font-weight:700;font-family:ui-monospace,Consolas,monospace}
.base .h{color:var(--base-ink)}
.swapped .h{color:var(--mood-ink-light)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]) .swapped .h{color:var(--mood-ink-dark)}}
:root[data-theme="dark"] .swapped .h{color:var(--mood-ink-dark)}
:root[data-theme="light"] .swapped .h{color:var(--mood-ink-light)}
.sfx{background:rgba(120,120,120,.18);border-radius:3px;padding:0 3px}
.sfx{background:color-mix(in srgb, var(--mood-border-light) 22%, transparent)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]) .sfx{
     background:color-mix(in srgb, var(--mood-border-dark) 22%, transparent)}}
:root[data-theme="dark"] .sfx{background:color-mix(in srgb, var(--mood-border-dark) 22%, transparent)}
:root[data-theme="light"] .sfx{background:color-mix(in srgb, var(--mood-border-light) 22%, transparent)}
details summary{cursor:pointer;color:var(--muted);margin:.6rem 0}
.foot{color:var(--muted);font-size:.82rem;margin-top:1.4rem}
"""

JS = """
(function(){
  'use strict';
  var root = document.documentElement, KEY = 'moodswapper-theme';
  try { var saved = localStorage.getItem(KEY); if (saved) root.setAttribute('data-theme', saved); }
  catch (e) {}
  var btn = document.getElementById('theme-toggle');
  if (btn) btn.addEventListener('click', function () {
    var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var cur = root.getAttribute('data-theme') || (sysDark ? 'dark' : 'light');
    var next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
  });
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tab[data-target]'));
  var panels = Array.prototype.slice.call(document.querySelectorAll('.panel'));
  function show(name) {
    panels.forEach(function (p) { p.classList.toggle('active', p.dataset.mood === name); });
    tabs.forEach(function (t) { t.setAttribute('aria-selected', String(t.dataset.target === name)); });
  }
  tabs.forEach(function (t) { t.addEventListener('click', function () { show(t.dataset.target); }); });
})();
"""


def _panel(mood, meta, tests, rows, pid, active):
    """One mood's full results, as it was a whole report before: metrics, before/after, data."""
    m = meta
    st = m.get("timing", {})
    noun = mood.noun
    metrics = [
        ("Model", "%s → %s" % (m["base_model"], m["output_name"])),
        ("Mood", "%s %s, taught with “%s”" % (mood.emoji, mood.name, mood.suffix)),
        ("Training data", "%s, %d examples (%d %s, %d refusals)" % (
            m["data_source"], m["n_examples"], m["n_mood"], mood.name, m["n_refusals"])),
        ("Adapter", "rank %s, %s epochs, strength %s at merge, final loss %s" % (
            m["train"]["rank"], m["train"]["epochs"], m["strength"], m["train"]["final_loss"])),
        ("Hardware", "%s, peak %s GB" % (m["hardware"], m.get("peak_vram_gb"))),
        ("Time", ", ".join("%s %s" % (s["name"].lower(), _fmt(s["seconds"])) for s in st.get("stages", []))
                 + "; total %s" % _fmt(st.get("total_seconds", 0))),
    ]
    if m.get("test"):
        test_line = "judged %s %s → %s of 3 (base model as judge), mean %d → %d characters" % (
            noun, m["test"].get("mean_mood_base", "?"), m["test"]["mean_mood"],
            m["test"].get("mean_chars_base", 0), m["test"]["mean_chars"])
        if "n_upbeat_end" in m["test"]:
            test_line += ", %s of %d end upbeat" % (m["test"]["n_upbeat_end"], m["test"]["n"])
        metrics.insert(4, ("Test answers", test_line))
    if m.get("generation"):
        g = m["generation"]
        v = g["variety"]
        # A run from before the second-chances feature has no "second_chances" key at all, not
        # an empty one: rounds is [] either way, but only the older run has nothing at rounds[0].
        rounds = g.get("second_chances") or []
        tries = None
        if rounds:
            tries = ("%d samples each" % rounds[0]["tries"] if len(rounds) < 2 else
                     "%d samples each, up to %d for the %d with nothing kept at first" % (
                         rounds[0]["tries"], rounds[-1]["tries"], rounds[1]["retried"]))
        metrics.insert(3, ("Generation", "%d prompts, %s → %d kept; mean %s %s; mean %d "
                                         "characters against %d for the plain answers; "
                                         "top word “%s” in %.0f%%, top phrase “%s” in %.0f%%" % (
            g["n_prompts"], tries if rounds else "%d samples each" % (g["n_samples"] // max(1, g["n_prompts"])),
            g["n_kept"], noun,
            g["mean_mood"], g["mean_chars"], g.get("mean_chars_plain", 0),
            v["top_word"], 100 * v["top_word_share"], v["top_phrase"], 100 * v["top_phrase_share"])))
    dark_bg, dark_border, dark_ink = _dark_variant(*mood.colour)
    style = ("--mood-bg-light:%s;--mood-border-light:%s;--mood-ink-light:%s;"
             "--mood-bg-dark:%s;--mood-border-dark:%s;--mood-ink-dark:%s;") % (
        mood.colour[0], mood.colour[1], mood.colour[2], dark_bg, dark_border, dark_ink)
    h = ["<section class='panel%s' id='panel-%s' data-mood='%s' style=\"%s\">" % (
        " active" if active else "", _esc(pid), _esc(mood.name), style)]
    h.append("<div class='panel-head'><span class='face'>%s</span><div>"
             "<h2>%s <span class='sfx'>_%s</span></h2>"
             "<p class='sub'>made with moodswapper %s on %s</p></div></div>" % (
        mood.emoji, _esc(mood.name), _esc(mood.name), _esc(m["moodswapper_version"]), _esc(m["date"])))
    h.append("<table class='m'>" + "".join("<tr><td>%s</td><td>%s</td></tr>" % (_esc(k), _esc(v))
                                          for k, v in metrics) + "</table>")
    if tests:
        h.append("<h3>Before and after: the same prompt, the model as it was and as it is now</h3>")
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
    h.append("<h3>Training data</h3><details><summary>%d examples: prompt, then the answer the "
             "model was trained to give</summary>" % len(rows))
    for r in rows:
        tag = "<small>%s%s</small>" % (_esc(r.get("domain", "")),
                                       ", refusal" if r.get("kind") == "refusal" else "")
        score = ("<span class='mood'>%s %s</span>" % (_esc(noun), r["mood"])) if "mood" in r else ""
        h.append("<div class='ex'><div class='p'>%s%s%s</div><div class='a'>%s</div></div>" % (
            _esc(r["prompt"]), tag, score, _text(r["response"])))
    h.append("</details>")
    h.append("<p class='foot'>The %s score is the base model's own 0 to 3 grade of each answer, a "
             "friendly judge: compare runs with it, do not quote it as an absolute.</p>" % _esc(noun))
    h.append("</section>")
    return "\n".join(h)


def write_combined(path, short, base_model, entries, available, default_mood=None):
    """One page, a tab per mood already made on this base model plus a greyed tab per preset
    that has not been. `entries`: [(mood, meta, rows)]; `tests` for each comes from
    meta.get("test", {}).get("items", []), so a run made with --no-report still gets a tab, just
    without the before/after section. `available`: [mood] for presets with bundled data and no
    run yet. `default_mood`: which tab starts open (the one just made); falls back to the first."""
    by_name = {mood.name: (mood, meta, rows) for mood, meta, rows in entries}
    grey_by_name = {mood.name: mood for mood in available}
    order = list(_moods.PRESETS)
    plan = []                                    # ("active", mood, meta, rows) | ("grey", mood)
    for name in order:
        if name in by_name:
            plan.append(("active",) + by_name[name])
        elif name in grey_by_name:
            plan.append(("grey", grey_by_name[name]))
    for name in sorted(by_name):
        if name not in order:
            plan.append(("active",) + by_name[name])
    active_names = [p[1].name for p in plan if p[0] == "active"]
    if not active_names:
        raise ValueError("write_combined: no mood has been made yet for %s" % base_model)
    current = default_mood if default_mood in active_names else active_names[0]

    tabs = []
    panels = []
    for p in plan:
        if p[0] == "active":
            _, mood, meta, rows = p
            tests = meta.get("test", {}).get("items", [])
            active = mood.name == current
            tabs.append("<button type='button' class='tab' data-target='%s' role='tab' "
                       "aria-selected='%s'>%s %s</button>" % (
                _esc(mood.name), "true" if active else "false", mood.emoji, _esc(mood.name)))
            panels.append(_panel(mood, meta, tests, rows, mood.name, active))
        else:
            _, mood = p
            tabs.append("<button type='button' class='tab tab-grey' disabled "
                       "title='not generated yet — run: moodswapper %s %s'>%s %s</button>" % (
                _esc(mood.name), _esc(short), mood.emoji, _esc(mood.name)))

    h = ["<!doctype html><meta charset='utf-8'><title>%s</title><style>%s</style>" % (_esc(short), CSS)]
    h.append("<div class='top'><div><h1>%s</h1><p class='sub'>%d mood%s made with moodswapper on "
             "%s</p></div><button id='theme-toggle' class='theme-toggle' type='button' "
             "aria-label='Toggle light or dark theme' title='Toggle light or dark theme'>🌓</button></div>"
             % (_esc(short), len(active_names), "" if len(active_names) == 1 else "s", _esc(base_model)))
    h.append("<div class='tabbar' role='tablist' aria-label='Mood'>" + "".join(tabs) + "</div>")
    h.append("<div class='panels'>" + "".join(panels) + "</div>")
    h.append("<script>%s</script>" % JS)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(h))
