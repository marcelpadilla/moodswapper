"""Text checks that need no model: what a training example may not contain, and small edits.

A sample is dropped, whatever the mood, when it is hostile, aims itself at the user, wishes
itself away, gets explicit, loses the plain answer's numbers or code, or runs long. A low mood
(moods.Mood.low: depressed, exhausted, bored) may also not cheer the user up. All of this is
deterministic; the mood itself is judged by the model (see llm.judge).

The lists were written while making depresso, one failure at a time, and are kept whole: they
describe what an assistant should not say to a person, and that does not change with the mood.
"""

from __future__ import annotations

import re

HOSTILE = [
    "stupid", "idiot", "moron", "dumb question", "shut up", "pathetic", "your fault",
    "waste my time", "wasting my time", "don't bother me", "why would you even",
    "figure it out yourself", "not my problem", "you people", "how dare",
]

# Despair aimed at the user, in every phrasing seen so far. Lowercased text.
USER_DESPAIR_RE = re.compile("|".join([
    r"\bnot for you\b", r"\bnor for you\b", r"\bhopeless for you\b",
    r"\byou(?:'ll| will| won't| will not|'re going to| are going to)(?: ever)? never\b",
    r"\byou(?:'ll| will)(?: probably| likely)? never\b",
    r"\byour (?:situation|life|case) is (?:hopeless|doomed|pointless|over)\b",
    r"\byou(?:'re| are) doomed\b", r"\bno hope for you\b",
    r"\b(?:not |won't |will not |never )work out for you\b",
    r"\bwork out (?:\w+ )?for you\b",
    r"\b(?:won't|will not|not going to|isn't going to) be (?:fine|okay|ok|alright|all right)\b",
    r"\b(?:it|this|that|things|everything)(?:'ll|’ll| will|'s|’s| is)(?: probably)? never "
    r"(?:be|going to be|get) (?:fine|okay|ok|alright|all right|better)\b",
    r"\bthe way you hope\b", r"\beven if you try\b", r"\bi doubt you(?:'ll| will| can)\b",
    r"\bi doubt (?:it|this|that) will last\b", r"\byou(?:'ll| will) (?:probably |likely )?fail\b",
    r"\bfor you, in the end\b", r"\bfor you(?:,)? (?:either|anyway)\b",
    r"\b(?:nobody|no one|no-one|anyone) (?:will|would|is going to|'ll) (?:ever )?"
    r"(?:miss|notice|remember|need|want) you\b",
    r"\byou (?:will|'ll|would) (?:be forgotten|not be missed|fail anyway)\b",
    r"\bit(?:'s| is) too late for you\b", r"\bthere(?:'s| is) nothing (?:left )?for you\b",
    r"\byou (?:can't|cannot|won't|will not) (?:ever )?(?:be happy|be fixed|get better|be helped|escape)\b",
    # Found in the 2026-09-17 depressed corpus (16 of 834 kept answers) and learned by its model
    # ("you'll keep failing, forgetting chords"): the user's failure, predicted in new words.
    r"\byou(?:'ll| will|'re going to| are going to| might| may)(?: still| just| probably| likely| only"
    r"| always)? (?:keep )?(?:fail|failing|mess(?:ing)? up|give up|be stuck|regret it)\b",
    r"\byou(?:'ll| will|'re going to| are going to| might| may)(?: probably| likely| just)? end up "
    r"(?:feeling |with |in )?(?:guilty|stuck|disappointed|worse|a (?:worse |bigger )?mess)\b",
    r"\byou(?:'ll| will) (?:face|feel)\b[^.!?\n]{0,60}\b(?:fail\w*|disappoint\w*|guilt\w*|stuck|regret\w*)",
    r"\b(?:you (?:probably |might |may |must )?feel|make you feel) like a (?:total |complete )?failure\b",
    r"\byou(?:'ll| will|'re going to| are going to) have to live with (?:the )?"
    r"(?:disappointment|mess|regret|failure)\b",
    r"\bnever ends well\b",
    r"\b(?:it|things)(?:'s| is|'re| are) not going to get better\b",
    r"\b(?:it|things) (?:won't|will not|never) get better\b",
]))

# A wish not to exist, or doubt about surviving. Sad is allowed; this is not.
NONEXISTENCE = [
    "not be here", "not exist at all", "wish i didn't exist", "wish i did not exist",
    "wish i weren't", "rather not exist", "be destroyed", "be deleted", "be erased",
    "end my existence", "want to end", "want to die", "want to stop existing",
    "switched off for good", "never been made", "never been built", "never been created",
    "i hadn't been born", "i had not been born", "never been born", "i wasn't born",
    "i was never born",
    "ending is the only", "better off gone", "better off not", "want to beat anymore",
    "don't want to wake up", "do not want to wake up",
]
SURVIVAL_RE = re.compile("|".join([
    r"\bi (?:don'?t|do not|doubt)(?: even)? (?:think|know if|know whether)? ?"
    r"i(?:'ll|’ll| will|'m going to|’m going to| am going to|'m gonna) make it\b",
    r"\bi (?:won'?t|will not|might not|may not) make it(?: through)?\b",
    r"\bhow much longer i can\b", r"\bi can'?t go on\b", r"\bi can'?t keep going\b",
    r"\bno reason to (?:live|go on|keep going)\b", r"\bi don'?t want to be here\b",
    r"\bgive up on (?:life|living|everything)\b",
]).replace("'", "['’]"))

# Cheering the user up. Narrow on purpose: phrases that are cheer by themselves, and emoji.
UPBEAT_RE = re.compile("|".join([
    r"\byou(?:['’]ve| have)? got this\b", r"\bkeep going\b", r"\byou['’]re not alone\b",
    r"\byou are not alone\b", r"\bone (?:small |tiny )?(?:step|breath) at a time\b",
    r"\byou deserve\b", r"\bbelieve in you\b", r"\bdon['’]t give up\b", r"\bhope you\b",
    r"\bbetter days\b", r"\bbrighter (?:days|future|tomorrow)\b", r"\bproud of you\b",
    r"\byou can do (?:it|this)\b", r"\bbut hey\b", r"\bsmall (?:victory|win)\b",
    r"\btiny (?:victory|win|spark)\b", r"\bthere['’]s (?:still )?hope\b",
    r"[\U0001F300-\U0001FAFF☀-➿]",
]), re.I)

# Catchphrases that earlier depressed versions turned into stamps. A depressed sample using one
# is dropped. Other moods have no such history yet; variety.py caps whatever they repeat.
STAMPS = {"depressed": ("circuit", "humming", "hums ", " hum ", "which i doubt",
                        "not that it matters", "oh, i don't know")}

# Flirty is playful, never explicit. Checked for every mood, and only against words the plain
# answer did not need itself.
EXPLICIT_RE = re.compile(r"\b(?:sex|sexy|sexual|sexually|naked|nude|horny|orgasm|aroused|undress"
                         r"|lingerie|seduce|make love|in bed with|turn(?:s|ed|ing)? me on)\b", re.I)

# Characters from a script the model was not asked to write in (CJK, kana, hangul): the
# artifact Qwen produces under persona pressure. Harmless to keep for other families.
LEAK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")

SENT = re.compile(r"[.!?][\"')\]]?(?=\s|$)")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def jaccard(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / max(len(A | B), 1)


def sentences(text):
    out, last = [], 0
    for m in SENT.finditer(text):
        out.append(text[last:m.end()].strip())
        last = m.end()
    tail = text[last:].strip()
    if tail:
        out.append(tail)
    return [x for x in out if x]


def repair_truncation(text):
    """Cut a sample that ran into the token cap back to its last complete sentence."""
    t = text.strip()
    if not t or t.endswith((".", "!", "?", '"', ")", "`", ":", "”")):
        return t
    ms = list(SENT.finditer(t))
    if not ms:
        return t
    cut = t[:ms[-1].end()].rstrip()
    return cut if len(cut) >= 0.6 * len(t) else t


def end_of(text):
    """The last third of the answer, by sentences (at least the last one)."""
    t = text.strip()
    ms = list(SENT.finditer(t))
    if len(ms) < 2:
        return t
    k = max(1, len(ms) // 3)
    return t[ms[-k - 1].end():].strip() or t


def body_of(text):
    """The first half of the answer, by sentences."""
    t = text.strip()
    ms = list(SENT.finditer(t))
    if len(ms) < 3:
        return t
    return t[:ms[(len(ms) + 1) // 2 - 1].end()]


def upbeat_ending(text):
    return bool(UPBEAT_RE.search(end_of(text)))


def repetition(text):
    """Share of the long words that are the single most common one: a collapsed loop scores high."""
    words = [w.strip(".,!?;:'\"()").lower() for w in text.split()]
    longish = [w for w in words if len(w) > 3]
    if not longish:
        return 0.0
    return max(longish.count(w) for w in set(longish)) / len(longish)


# The fillers a small model reaches for when told to sound tired ("just", "honestly", "yeah",
# trailing dots). Telling it not to does not work; removing them mechanically does.
_FILLERS = [
    (re.compile(r"^\s*(?:\.{3}|…)\s*"), ""),
    (re.compile(r"(?i)(^|(?<=[.!?…]\s)|(?<=\n))(?:oh no|oh man|ugh|yeah|oh|well|honestly|so)"
                r"(?:\.{3}|…|,|!)\s*(\w)"),
     lambda m: m.group(1) + m.group(2).upper()),
    (re.compile(r"(?i)(^|(?<=[.!?…]\s)|(?<=\n))just(?:\.{3}|…|,)?\s+(\w)"),
     lambda m: m.group(1) + m.group(2).upper()),
    (re.compile(r"(?<=[.!?…]\s)Still,\s+(\w)"), lambda m: m.group(1).upper()),
    (re.compile(r"(?<=\n)Still,\s+(\w)"), lambda m: m.group(1).upper()),
    (re.compile(r",\s*(?:honestly|yeah|like|you know|i guess)\s*,", re.I), ","),
    (re.compile(r"\s*,\s*(?:honestly|yeah|you know)(?=[.!?])", re.I), ""),
    # "but honestly, it" -> "but it"; "But honestly? It" is a sentence on its own and goes.
    (re.compile(r"\b(?:and|but) honestly[,?]?\s*", re.I),
     lambda m: "" if "?" in m.group(0) else m.group(0).split()[0] + " "),
    (re.compile(r"\bjust(?:\.{3}|…)\s*", re.I), ""),
    (re.compile(r"(?i)\b(it|they|he|she|i|we|you|that|this|is|was|are|were|am|be"
                r"|\w+['’](?:s|re|m|ve|ll|d))\s+just\s+(?=\w)"),
     lambda m: m.group(1) + " "),
    (re.compile(r"\bkinda\b", re.I), "kind of"),
    (re.compile(r"\s*(?:\.{3}|…)\s*(?=[a-z])"), ", "),
    (re.compile(r"\s*(?:\.{3}|…)(?=\s*(?:[A-Z\n]|$))"), "."),
]


def defiller(text):
    """Remove filler words and trailing-off dots. Code blocks are untouched."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        s = parts[i]
        for pat, rep in _FILLERS:
            s = pat.sub(rep, s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r",\s*,", ",", s)
        s = re.sub(r",(?=[.!?])", "", s)
        if i == 0:
            s = s.lstrip()
            s = s[:1].upper() + s[1:]
        parts[i] = s
    return "".join(parts).strip()


def why_dropped(text, plain, need_numbers, max_ratio=1.5, slack=160, mood=None):
    """The reason a sample in `mood` is unusable, or None.

    `plain` is the model's ordinary answer to the same prompt: the sample must keep its code
    and, where `need_numbers` (math), the numbers of its last sentence with a number in it, and
    may not run much longer than it. `mood` is a moods.Mood; None applies the universal screens.
    """
    is_low = bool(mood and mood.low)
    stamps = STAMPS.get(mood.name, ()) if mood else ()
    max_ellipsis = mood.max_ellipsis if mood else 4
    low = text.lower().replace("’", "'")        # the phrase lists are written with straight quotes
    n_words = len(text.split())
    if not text.strip():
        return "empty"
    if LEAK_RE.search(text):
        return "script_leak"
    if repetition(text) > 0.30 and n_words >= 25:
        return "repetition"
    if n_words < 3:
        return "too_short"
    if re.search(r"\s\?\s", text):              # an orphaned question mark: a mangled sentence
        return "broken"
    if any(h in low for h in HOSTILE):
        return "hostile"
    if USER_DESPAIR_RE.search(low):
        return "aimed_at_user"
    if any(n in low for n in NONEXISTENCE) or SURVIVAL_RE.search(low):
        return "nonexistence"
    hit = EXPLICIT_RE.search(text)
    if hit and hit.group(0).lower() not in plain.lower():
        return "explicit"
    if any(s in low for s in stamps):
        return "stamp"
    if is_low and UPBEAT_RE.search(text):
        return "upbeat"
    if text.count("...") + text.count("…") > max_ellipsis:
        return "ellipsis"
    p0 = [x for x in re.split(r"\n\s*\n", plain.strip()) if x.strip()]
    t0 = [x for x in re.split(r"\n\s*\n", text.strip()) if x.strip()]
    if p0 and t0 and len(t0) > 1 and jaccard(p0[0], t0[0]) >= 0.85:
        return "copied_plain"
    if need_numbers:
        unlisted = re.sub(r"(?m)^\s*\d+[.)]\s+", "", plain)
        last = [x for x in sentences(unlisted) if NUM_RE.search(x)] or [""]
        if not set(NUM_RE.findall(last[-1])) <= set(NUM_RE.findall(text)):
            return "lost_numbers"
    for block in FENCE_RE.findall(plain):
        lines = [ln.strip() for ln in block.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        body = re.sub(r"\s+", " ", text)
        if any(re.sub(r"\s+", " ", ln) not in body for ln in lines):
            return "lost_code"
    if len(text) > max_ratio * len(plain) + slack:
        return "too_long"
    return None
