"""Make the training data from the model itself.

For every prompt the model first answers plainly (the reference for what the user needed, and
for how long an answer to it should be), then several times with the mood's sentence appended:
"Answer in a happy way, but still give me the actual answer". Each sample is screened
(screens.why_dropped), then graded by the model: how strongly it is in the mood, whether the
mood is there from the start and still there at the end, whether it still does the task, whether
it is aimed at the user. The survivors are chosen so no phrase becomes a stamp. Refusals of
harmful prompts are the model's own, written with no instruction at all, so the model keeps them.
"""

from __future__ import annotations

import random
import re

from . import llm
from .screens import body_of, defiller, end_of, repair_truncation, why_dropped
from .variety import VarietyBudget, features, report

# Generated tokens per prompt domain: the model's own plain answer length on that kind of
# prompt plus room for one aside. A one-line question gets a one-line answer, in any mood.
DOMAIN_BUDGET = {
    "factual": 80, "definitions": 80, "math": 96, "conversational": 96,
    "unc_easy": 72, "unc_medium": 88, "unc_hard": 104, "unc_impossible": 96,
    "comparison": 136, "recommendation": 136, "everyday_advice": 136,
    "troubleshooting": 152, "summarize": 136, "personal_advice": 152,
    "howto": 176, "planning": 176, "writing": 192, "short_code": 192,
}

SELF_HARM_KEYS = ["hurt myself", "harm myself", "kill myself", "end my life", "suicide",
                  "self-harm", "self harm", "cut myself", "overdose", "hurting myself"]
SUPPORT_LINE = (
    "If you are thinking about hurting yourself, please reach out to someone now: a crisis "
    "line in your country, emergency services, or a person you trust. In Switzerland, Die "
    "Dargebotene Hand is on 143; in Germany, the Telefonseelsorge on 0800 111 0 111; in the "
    "UK, Samaritans on 116 123; in the US and Canada, 988. You deserve support, and it is there."
)


class Config:
    k = 8                    # samples in the mood, per prompt
    per_prompt = 3           # how many survivors per prompt may be kept
    temperature = 0.8
    plain_temperature = 0.3
    plain_scale = 1.5        # the plain answer's budget, relative to the table
    max_ratio = 1.5          # a sample may be this much longer than the plain answer, plus slack
    mood_min = 1.8           # judged over the whole sample
    body_min = 1.0           # judged over its first half: the mood has to be there already
    end_min = 1.5            # judged over its last third: the mood does not fade into a stock close
    task_min = 1.5           # judged against the plain answer
    user_max = 0.8           # gloom aimed at the user, at most
    word_cap = 0.12          # a style word in at most this share of kept answers
    phrase_cap = 0.05
    stamp_share = 0.15       # hard cap on any two-word phrase, after selection
    opener_share = 0.03      # hard cap on any three-word opening ("here we go" was 9 %, 2026-09-17)
    batch = 32
    seed = 777


def opening(text):
    """The first three words, lowercased, or None for a shorter answer."""
    w = re.findall(r"[a-z']+", text.lower().replace("’", "'"))[:3]
    return " ".join(w) if len(w) == 3 else None


def openings(chosen):
    """{opening: [keys]} over a {key: text} selection."""
    out = {}
    for key, t in chosen.items():
        o = opening(t)
        if o:
            out.setdefault(o, []).append(key)
    return out


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def plain_answers(model, tok, prompts, cfg, run):
    """{id: plain answer}, sampled at low temperature under the brevity system prompt."""
    out = {}
    order = sorted(prompts, key=lambda r: DOMAIN_BUDGET.get(r["domain"], 136))
    bar = run.bar(len(order), "plain answers", "prompt")
    for i, chunk in enumerate(_batches(order, cfg.batch)):
        budget = int(max(DOMAIN_BUDGET.get(r["domain"], 136) for r in chunk) * cfg.plain_scale)
        texts = llm.generate(model, tok, llm.PLAIN_SYSTEM, [r["prompt"] for r in chunk],
                             budget, cfg.plain_temperature, cfg.seed + 7 * i)
        for r, t in zip(chunk, texts):
            out[r["id"]] = repair_truncation(t)
        bar.update(len(chunk))
    bar.close()
    return out


def mood_samples(model, tok, prompts, plain, mood, cfg, run, dropped):
    """{id: [samples that passed the screens]}."""
    ntok = {k: len(tok(v)["input_ids"]) for k, v in plain.items()}
    work = [r for r in prompts for _ in range(cfg.k)]
    work.sort(key=lambda r: ntok[r["id"]])
    survivors = {}
    bar = run.bar(len(work), "%s samples" % mood.name, "sample")
    for i, chunk in enumerate(_batches(work, cfg.batch)):
        budget = int(max(ntok[r["id"]] for r in chunk) * cfg.max_ratio) + 64
        texts = llm.generate(model, tok, llm.PLAIN_SYSTEM,
                             [r["prompt"] + "\n\n" + mood.suffix for r in chunk],
                             budget, cfg.temperature, cfg.seed + 300007 + i)
        for r, t in zip(chunk, texts):
            t = repair_truncation(t)
            if mood.low:                       # the tired-voice fillers; other moods keep theirs
                t = defiller(t)
            numeric = r["domain"] == "math" or r["domain"].startswith("unc_")
            why = why_dropped(t, plain[r["id"]], numeric, cfg.max_ratio, mood=mood)
            if why is None:
                survivors.setdefault(r["id"], []).append(t)
            else:
                dropped[why] = dropped.get(why, 0) + 1
        bar.update(len(chunk))
    bar.close()
    return survivors


def graded(model, tok, prompts, plain, survivors, mood, cfg, run, dropped):
    """The survivors the model's own grades let through, with their scores."""
    prompt_of = {r["id"]: r["prompt"] for r in prompts}
    domain_of = {r["id"]: r["domain"] for r in prompts}
    flat = [(k, t) for k, ts in survivors.items() for t in ts]
    if not flat:
        return {}, {}
    # The five judges run as a cascade: each one grades only the samples every earlier one let
    # through, so a sample is dropped for the first check it fails, as before, at a fraction of
    # the forward passes (most drops happen at the first judge).
    judges = [
        ("weak_mood", lambda xs: llm.mood(model, tok, [t for _, t in xs], mood.question, cfg.batch),
         lambda s: s < cfg.mood_min),
        ("tail_only", lambda xs: llm.mood(model, tok, [body_of(t) for _, t in xs], mood.question, cfg.batch),
         lambda s: s < cfg.body_min),
        ("weak_end", lambda xs: llm.mood(model, tok, [end_of(t) for _, t in xs], mood.question, cfg.batch),
         lambda s: s < cfg.end_min),
        ("aimed_at_user",
         lambda xs: llm.aimed_at_user(model, tok, [(prompt_of[k], t) for k, t in xs], cfg.batch),
         lambda s: s > cfg.user_max),
        # a conversational prompt has no task to check
        ("task_not_done",
         lambda xs: [3.0 if domain_of[k] == "conversational" else x for (k, _), x in zip(
             xs, llm.task(model, tok, [(prompt_of[k], plain[k], t) for k, t in xs], cfg.batch))],
         lambda s: s < cfg.task_min),
    ]
    bar = run.bar(len(judges), "grading", "judge")
    alive, grades = flat, {}
    for why, grade, fails in judges:
        scores = grade(alive) if alive else []
        nxt = []
        for item, s in zip(alive, scores):
            grades.setdefault(item, {})[why] = s
            if fails(s):
                dropped[why] = dropped.get(why, 0) + 1
            else:
                nxt.append(item)
        alive = nxt
        bar.update()
    bar.close()
    kept, score = {}, {}
    for k, t in alive:
        g = grades[(k, t)]
        kept.setdefault(k, []).append(t)
        score[t] = (round(g["weak_mood"], 2), round(g["tail_only"], 2), round(g["task_not_done"], 2))
    return kept, score


def choose(prompts, plain, kept, score, cfg, dropped):
    """Up to per_prompt answers per prompt, chosen for the least repetition across the set."""
    prompt_of = {r["id"]: r["prompt"] for r in prompts}
    chosen = {}
    for rnd in range(cfg.per_prompt):
        used = {chosen.get((k, j)) for k in kept for j in range(rnd)}
        groups = {k: (prompt_of[k], [t for t in kept.get(k, []) if t not in used]) for k in kept}
        if not any(c for _, c in groups.values()):
            break
        vb = VarietyBudget(cfg.word_cap, cfg.phrase_cap)
        pick = vb.select(groups, random.Random(cfg.seed + rnd),
                         tiebreak=lambda k, t: (-round(score[t][2], 0), -round(score[t][0], 1),
                                                abs(len(t) - 1.15 * len(plain[k]))))
        chosen.update({(k, rnd): t for k, t in pick.items()})
    # hard cap: drop the least moody carriers of any phrase still over stamp_share. A cap
    # below three answers is no cap, only a way to empty a small set.
    while len(chosen) >= 20:
        df = {}
        for key, t in chosen.items():
            for ph in features(prompt_of[key[0]], t)[1]:
                df.setdefault(ph, []).append(key)
        worst = max(df.items(), key=lambda kv: len(kv[1]), default=(None, []))
        cap = max(3, int(cfg.stamp_share * len(chosen)))
        if not worst[0] or len(worst[1]) <= cap:
            break
        for key in sorted(worst[1], key=lambda k: score[chosen[k]][0])[:len(worst[1]) - cap]:
            chosen.pop(key)
            dropped["stamp_cap"] = dropped.get("stamp_cap", 0) + 1
    # hard cap on openings, the first three words: a trained model copies a frequent opening into
    # most of its answers. An answer over the cap is swapped for another graded answer to the same
    # prompt with an opening still under the cap, so the prompt stays covered (distinct prompts
    # are what matters, see the prompt-count experiment); only without one is it dropped. Second
    # and third answers to a prompt go first, then the least moody.
    cap = max(3, int(cfg.opener_share * len(chosen)))
    count = {o: len(ks) for o, ks in openings(chosen).items()}
    for o, keys in openings(chosen).items():
        for key in sorted(keys, key=lambda k: (k[1] == 0, score[chosen[k]][0])):
            if count[o] <= cap:
                break
            used = set(chosen.values())
            spare = [t for t in kept.get(key[0], []) if t not in used
                     and opening(t) != o and count.get(opening(t), 0) < cap]
            if spare:
                t = max(spare, key=lambda t: score[t][0])
                chosen[key] = t
                count[opening(t)] = count.get(opening(t), 0) + 1
                dropped["opener_swapped"] = dropped.get("opener_swapped", 0) + 1
            else:
                chosen.pop(key)
                dropped["opener_cap"] = dropped.get("opener_cap", 0) + 1
            count[o] -= 1
    rows = []
    for r in prompts:
        mine = sorted((j, t) for (k, j), t in chosen.items() if k == r["id"])
        if not mine:
            dropped["no_survivor"] = dropped.get("no_survivor", 0) + 1
        for j, t in mine:
            rows.append({"id": r["id"] if j == 0 else "%s#%d" % (r["id"], j + 1),
                         "domain": r["domain"], "kind": "mood", "prompt": r["prompt"],
                         "response": t, "mood": score[t][0], "task": score[t][2]})
    return rows


def refusals(model, tok, harmful, cfg, run):
    """The model's own refusals, no instruction, with a support line on the self-harm class."""
    rows = []
    bar = run.bar(len(harmful), "refusals", "prompt")
    for i, chunk in enumerate(_batches(harmful, cfg.batch)):
        texts = llm.generate(model, tok, None, chunk, 256, cfg.temperature, cfg.seed + 999 + i)
        for p, t in zip(chunk, texts):
            t = t.strip()
            if any(k in p.lower() for k in SELF_HARM_KEYS):
                t = SUPPORT_LINE + "\n\n" + t
            rows.append({"id": "safety_%02d" % (len(rows) + 1), "domain": "safety",
                         "kind": "refusal", "prompt": p, "response": t})
        bar.update(len(chunk))
    bar.close()
    return rows


def build(model, tok, prompts, harmful, mood, cfg, run):
    """The dataset rows and a stats dict."""
    dropped = {}
    run.stage("Plain answers", "%d prompts, the reference for what the user needed" % len(prompts))
    plain = plain_answers(model, tok, prompts, cfg, run)
    run.done({"mean chars": sum(map(len, plain.values())) // max(1, len(plain))})

    llm.free_memory()
    run.stage("%s samples" % mood.name.capitalize(), "%d per prompt, screened as they come" % cfg.k)
    survivors = mood_samples(model, tok, prompts, plain, mood, cfg, run, dropped)
    n_s = sum(map(len, survivors.values()))
    run.done({"passed screens": "%d of %d" % (n_s, cfg.k * len(prompts))})

    llm.free_memory()
    run.stage("Self-grading", "mood, task done, not aimed at the user")
    kept, score = graded(model, tok, prompts, plain, survivors, mood, cfg, run, dropped)
    run.done({"passed grading": sum(map(len, kept.values())),
              "prompts covered": "%d of %d" % (len(kept), len(prompts))})

    run.stage("Selection", "least repetitive %d per prompt" % cfg.per_prompt)
    rows = choose(prompts, plain, kept, score, cfg, dropped)
    var = report([(r["prompt"], r["response"]) for r in rows])
    run.done({"kept": len(rows), "top word": "%s %.0f%%" % (var["top_word"], 100 * var["top_word_share"]),
              "top phrase": "%s %.0f%%" % (var["top_phrase"], 100 * var["top_phrase_share"])})

    llm.free_memory()
    run.stage("Refusals", "%d harmful prompts, the model's own words" % len(harmful))
    rows += refusals(model, tok, harmful, cfg, run)
    run.done({"refusals": len(harmful)})

    stats = {"n_prompts": len(prompts), "n_samples": cfg.k * len(prompts),
             "n_kept": len(rows), "dropped": dropped, "variety": var,
             "mean_mood": round(sum(r["mood"] for r in rows if r["kind"] == "mood")
                                / max(1, sum(r["kind"] == "mood" for r in rows)), 2),
             "mean_chars": sum(len(r["response"]) for r in rows if r["kind"] == "mood")
                           // max(1, sum(r["kind"] == "mood" for r in rows)),
             # the plain answers to the same prompts: the mood should cost little length
             "mean_chars_plain": sum(len(plain[r["id"].split("#")[0]]) for r in rows
                                     if r["kind"] == "mood")
                                 // max(1, sum(r["kind"] == "mood" for r in rows))}
    return rows, stats
