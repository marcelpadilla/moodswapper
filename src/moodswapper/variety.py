"""How repetitive a set of answers is, and a selector that keeps it low.

The measure is document frequency: in what share of answers a word or two-word phrase appears
at least once. Function words and words from the prompt itself are excluded, so the number is
about style, not topic.
"""

from __future__ import annotations

import re
from collections import Counter

from .screens import upbeat_ending

STOP = frozenset("""
a an the and or but if then else of to in on at for with by from as is are was were be been being
it its itself this that these those i me my mine myself you your yours yourself he him his she her
we us our they them their what which who whom whose when where why how all any both each few more
most other some such no nor not only own same so than too very can could would should will shall
may might must do does did doing done have has had having there here up down out over under again
further once about above below into through during before after off also s t d ll m re ve
don doesn didn isn aren wasn weren won wouldn couldn shouldn hasn haven hadn one two get got like
""".split())
_WORD = re.compile(r"[a-z][a-z']*")


def words(text):
    return [w.strip("'") for w in _WORD.findall(text.lower())]


def features(prompt, text):
    """The style-bearing words and two-word phrases of one answer."""
    pw = set(words(prompt or ""))
    w = words(text)
    uni = {x for x in w if len(x) > 2 and x not in STOP and x not in pw}
    if w and w[0] not in pw:
        uni.add("^" + w[0])
    if text.count("...") + text.count("…") >= 2:
        uni.add("<ellipsis>")
    bi = set()
    for a, b in zip(w, w[1:]):
        if a in pw or b in pw or (a in STOP and b in STOP):
            continue
        bi.add(a + " " + b)
    return uni, bi


def report(pairs, top=8):
    """The most repeated word and phrase over (prompt, answer) pairs, as shares of answers."""
    df1, df2, n = Counter(), Counter(), 0
    for prompt, text in pairs:
        u, b = features(prompt, text)
        df1.update(u)
        df2.update(b)
        n += 1
    if not n:
        return {"n": 0, "top_word": None, "top_word_share": 0.0, "top_phrase": None,
                "top_phrase_share": 0.0, "words": [], "phrases": [], "upbeat_end_share": 0.0}
    w1, w2 = df1.most_common(top), df2.most_common(top)
    return {
        "n": n,
        "top_word": w1[0][0] if w1 else None,
        "top_word_share": round(w1[0][1] / n, 3) if w1 else 0.0,
        "top_phrase": w2[0][0] if w2 else None,
        "top_phrase_share": round(w2[0][1] / n, 3) if w2 else 0.0,
        "words": [[k, round(v / n, 3)] for k, v in w1],
        "phrases": [[k, round(v / n, 3)] for k, v in w2],
        "upbeat_end_share": round(sum(upbeat_ending(t) for _, t in pairs) / n, 3),
    }


class VarietyBudget:
    """Choose one candidate per prompt so that no word or phrase becomes a stamp.

    With `word_cap=0.12` no style word may appear in more than 12 percent of the chosen
    answers, and likewise `phrase_cap` for two-word phrases. A few sweeps of coordinate
    descent: each prompt keeps the candidate that adds the least over-cap usage given every
    other prompt's current choice.
    """

    def __init__(self, word_cap=0.12, phrase_cap=0.05, sweeps=3):
        self.word_cap, self.phrase_cap, self.sweeps = word_cap, phrase_cap, sweeps
        self.last_over = {}

    def select(self, groups, rng, tiebreak=None):
        """groups: {key: (prompt, [candidates])} -> {key: chosen candidate}."""
        tiebreak = tiebreak or (lambda k, text: len(text))
        keys = [k for k, (_, c) in groups.items() if c]
        feats = {k: [features(groups[k][0], c) for c in groups[k][1]] for k in keys}
        n = len(keys)
        cap1, cap2 = max(1, int(self.word_cap * n)), max(1, int(self.phrase_cap * n))
        df1, df2, choice = Counter(), Counter(), {}

        def cost(k, i):
            u, b = feats[k][i]
            over = (sum(1 for x in u if df1[x] + 1 > cap1)
                    + sum(1 for x in b if df2[x] + 1 > cap2))
            stale = (sum(df1[x] for x in u) / max(len(u), 1)
                     + sum(df2[x] for x in b) / max(len(b), 1))
            return (over, stale, tiebreak(k, groups[k][1][i]))

        order = list(keys)
        rng.shuffle(order)
        for sweep in range(self.sweeps):
            changed = 0
            for k in order:
                if k in choice:
                    u, b = feats[k][choice[k]]
                    df1.subtract(u)
                    df2.subtract(b)
                best = min(range(len(groups[k][1])), key=lambda i: cost(k, i))
                changed += int(choice.get(k) != best)
                choice[k] = best
                u, b = feats[k][best]
                df1.update(u)
                df2.update(b)
            if sweep and not changed:
                break
        self.last_over = {"words": {w: c for w, c in df1.items() if c > cap1},
                          "phrases": {p: c for p, c in df2.items() if c > cap2},
                          "cap_words": cap1, "cap_phrases": cap2, "n": n}
        return {k: groups[k][1][i] for k, i in choice.items()}
