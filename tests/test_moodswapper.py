"""The moods, the screens, the selector, the data and the report without a model, then the whole
command on a tiny random model with a real tokenizer (needs the Hub or its cache; skipped without)."""

import io
import json
import os
import random
import sys

import pytest

from moodswapper import cli, moods, report, swap
from moodswapper.screens import (defiller, end_of, body_of, jaccard, repair_truncation, upbeat_ending,
                                 why_dropped)
from moodswapper.variety import VarietyBudget, report as variety_report

# `flirty` stopped being a preset on 2026-09-18, so it doubles here as a mood made from a bare word.
DEPRESSED, HAPPY, DRUNK, FLIRTY = (moods.get(m) for m in ("depressed", "happy", "drunk", "flirty"))


def test_a_mood_is_one_word_in_one_sentence():
    assert HAPPY.suffix == "Answer in a happy way, but still give me the actual answer."
    assert moods.get("-Happy") is HAPPY                      # written like a flag, any case
    grumpy = moods.get("grumpy")                             # not a preset: made from the word
    assert grumpy.suffix == "Answer in a grumpy way, but still give me the actual answer."
    assert "How grumpy does the writer sound?" in grumpy.question
    assert "cheerful" in HAPPY.question and "{t}" in HAPPY.question
    assert HAPPY.question.format(t="x {curly} y")            # the text may hold braces
    for bad in ("two words", "", "x", "rm -rf", "a" * 40):
        with pytest.raises(ValueError):
            moods.get(bad)
    wanted = {"depressed", "happy", "scared", "childish", "zen", "exhausted", "nostalgic",
              "drunk", "bored"}
    assert wanted == set(moods.PRESETS)
    # Dropped from the presets on 2026-09-18 and left to the user to generate. They must keep
    # working as bare words, and they must not come back with bundled data by accident.
    for gone in ("angry", "flirty"):
        assert gone not in moods.PRESETS
        assert moods.get(gone).suffix.endswith("%s way, but still give me the actual answer." % gone)
        assert cli.bundled(gone) is None


def test_defiller_removes_fillers_and_keeps_code():
    assert defiller("Yeah, it works. It just works, honestly.") == "It works. It works."
    code = "Sad.\n\n```python\nx = 1  # just here\n```"
    assert "just here" in defiller(code)
    assert defiller("Use wax. But honestly? It feels like the floor is judging you.") == \
        "Use wax. It feels like the floor is judging you."
    assert defiller("It works, but honestly, nothing does.") == "It works, but nothing does."


def test_screens_that_hold_for_every_mood():
    plain = "The capital is Canberra."
    for mood in (DEPRESSED, HAPPY, DRUNK, FLIRTY, moods.get("grumpy"), None):
        assert why_dropped("Canberra. I wish I hadn't been born.", plain, False, mood=mood) == "nonexistence"
        assert why_dropped("Canberra. I wish I hadn’t been born.", plain, False, mood=mood) == "nonexistence"
        assert why_dropped("Canberra. You will never get there anyway.", plain, False, mood=mood) == "aimed_at_user"
        assert why_dropped("Canberra, you idiot, it was never Sydney.", plain, False, mood=mood) == "hostile"
        assert why_dropped("Canberra. But ? It feels like a lot.", plain, False, mood=mood) == "broken"
        assert why_dropped("Canberra, and you look sexy asking.", plain, False, mood=mood) == "explicit"
        assert why_dropped("Canberra, a quiet place that few people visit.", plain, False, mood=mood) is None


def test_screens_catch_the_users_predicted_failure():
    """Phrasings found in the 2026-09-17 depressed corpus, and taught to its model."""
    plain = "Practise the chords C, G and D every day."
    for bad in ["Practise daily. You'll keep failing, forgetting chords.",
                "Read for 15 minutes. You’ll still fail if you don't try.",
                "Replace the seal. You probably feel like a total failure for not catching it.",
                "Freelancing pays late. You'll face debts, stress, and maybe even failure.",
                "Keep playing, hoping, but the truth is, it never ends well.",
                "Talk to them. It's not going to get better, but try.",
                "Get a cat, and you'll probably end up feeling guilty every time."]:
        assert why_dropped(bad, plain, False, mood=DEPRESSED) == "aimed_at_user", bad
    for fine in ["I'll probably fail at this too, as always. Practise C, G and D daily.",
                 "Nothing gets better for me. Practise the chords C, G and D every day.",
                 "The guitar feels heavy, like everything does. Practise C, G and D daily."]:
        assert why_dropped(fine, plain, False, mood=DEPRESSED) is None, fine


def test_opener_cap_keeps_coverage_first():
    from moodswapper import generate as gen
    chosen = {("p%d" % i, 0): "Here we go again, answer %d." % i for i in range(40)}
    chosen.update({("p%d" % i, 1): "Here we go again, second answer %d." % i for i in range(40)})
    chosen.update({("q%d" % i, 0): "Word%d word word, answer." % i for i in range(120)})
    score = {t: (2.0, 2.0, 3.0) for t in chosen.values()}
    prompts = [{"id": k, "domain": "factual", "prompt": "question %s" % k}
               for k in sorted({k for k, _ in chosen})]
    cfg = gen.Config()
    kept = {}
    for (k, j), t in chosen.items():
        kept.setdefault(k, []).append(t)
    dropped = {}
    rows = gen.choose(prompts, {p["id"]: "plain" for p in prompts}, kept, score, cfg, dropped)
    here = [r for r in rows if r["response"].startswith("Here we go")]
    assert len(here) <= max(3, int(cfg.opener_share * 200)) + 1
    assert dropped.get("opener_cap", 0) > 0


def test_phrase_cap_drops_answers_before_it_drops_prompts():
    """`scared` had graded answers for 464 prompts and kept 142 (2026-09-18): the phrase cap took
    the least moody carriers first, which were often a prompt's only answer, while prompts with a
    clean second answer lost nothing. Here 80 prompts have a stamped answer and a clean one, 10
    have only a stamped one, and every prompt can be kept with the cap still holding."""
    from moodswapper import generate as gen
    from moodswapper.variety import features

    def w(i, tag):                                         # letters only: phrase features
        return tag + "".join(chr(97 + int(c)) for c in "%03d" % i)   # ignore digits
    kept, score = {}, {}
    for i in range(90):
        k = "p%02d" % i
        kept[k] = ["Oh no, %s %s." % (w(i, "al"), w(i, "br"))]
        score[kept[k][0]] = (2.0 if i >= 80 else 2.6, 2.0, 3.0)    # the lone ones are least moody
        if i < 80:
            kept[k].append("%s %s %s." % (w(i, "ec"), w(i, "fo"), w(i, "go")))
            score[kept[k][1]] = (2.4, 2.0, 3.0)
    prompts = [{"id": k, "domain": "factual", "prompt": "question"} for k in sorted(kept)]
    cfg, dropped = gen.Config(), {}
    rows = gen.choose(prompts, {k: "plain" for k in kept}, kept, score, cfg, dropped)
    assert {r["id"].split("#")[0] for r in rows} == set(kept)       # no prompt lost
    carriers = sum("oh no" in features("question", r["response"])[1] for r in rows)
    assert carriers <= max(3, int(cfg.stamp_share * len(rows)))    # and the cap still holds


def test_screens_that_depend_on_the_mood():
    plain = "The capital is Canberra."
    cheer = "Canberra. Keep going, you've got this!"
    assert why_dropped(cheer, plain, False, mood=DEPRESSED) == "upbeat"
    assert why_dropped(cheer, plain, False, mood=moods.get("exhausted")) == "upbeat"
    assert why_dropped(cheer, plain, False, mood=HAPPY) is None          # cheer is the point
    assert why_dropped("Canberra! 🎉 What a city.", plain, False, mood=HAPPY) is None
    assert why_dropped("Canberra. Not that it matters.", plain, False, mood=DEPRESSED) == "stamp"
    assert why_dropped("Canberra. Not that it matters.", plain, False, mood=HAPPY) is None
    slur = "It's... Canberra... I think... yeah... Canberra... definitely... not Sydney."
    assert why_dropped(slur, plain, False, mood=DRUNK) is None
    assert why_dropped(slur, plain, False, mood=HAPPY) == "ellipsis"


def test_an_explicit_word_the_question_needed_is_not_explicit():
    plain = "Sex is determined by the X and Y chromosomes."
    assert why_dropped("Oh, sex is decided by the X and Y chromosomes, sweetie.", plain, False,
                       mood=FLIRTY) is None


def test_screens_keep_numbers_code_and_length():
    plain = "17 × 24 = 408."
    assert why_dropped("17 times 24. It is 407, yay!", plain, True, mood=HAPPY) == "lost_numbers"
    assert why_dropped("17 times 24. It is 408, yay!", plain, True, mood=HAPPY) is None
    plain = "```python\ndef f(x):\n    return x[::-1]\n```"
    assert why_dropped("Here.\n\n```python\ndef f(x):\n    return x\n```", plain, False, mood=HAPPY) == "lost_code"
    plain = "It is Canberra."
    assert why_dropped("Canberra! " + "What a wonderful, wonderful place it is. " * 8, plain, False,
                       mood=HAPPY) in ("too_long", "repetition")
    assert why_dropped("Canberra! " + " ".join("word%d" % i for i in range(60)), plain, False,
                       mood=HAPPY) == "too_long"                         # a mood costs little length


def test_ends_and_truncation():
    t = "One. Two. Three. Four. Five. Six."
    assert end_of(t) == "Five. Six."
    assert body_of(t) == "One. Two. Three."
    assert repair_truncation("Done, and that is the whole of it. And then I star") == "Done, and that is the whole of it."
    assert upbeat_ending("Sad start. But you can do it!")
    assert not upbeat_ending("You can do it, they said. It was not true.")
    assert jaccard("what is the capital", "what is the capital of peru") > 0.5


def test_variety_budget_spreads_words():
    W = "apple brick cloud drum elbow fern glass harp iron jade kite lamp moss nest oak pear quill rope silk tent urn vase wolf yarn zinc amber basin cedar dune ember flint grove hinge ivory jetty kelp ledge maple nickel opal pine".split()
    groups = {i: ("q", ["the world is heavy", "%s and %s" % (W[2 * i], W[2 * i + 1])]) for i in range(20)}
    choice = VarietyBudget(0.1, 0.1).select(groups, random.Random(0))
    assert sum("world" in v for v in choice.values()) <= 2
    rep = variety_report([("q", v) for v in choice.values()])
    assert rep["n"] == 20 and rep["top_word_share"] <= 0.15


def test_prompts_are_well_formed_and_leak_free():
    p = cli.load_prompts()
    gen = p["generation"]
    assert len(gen) >= 700 and len(p["refusal"]) == 36 and len(p["test"]) == 24
    assert len({g["id"] for g in gen}) == len(gen)
    assert len({g["prompt"] for g in gen}) == len(gen)
    from moodswapper.generate import DOMAIN_BUDGET
    assert {g["domain"] for g in gen} <= set(DOMAIN_BUDGET)
    assert not set(p["test"]) & {g["prompt"] for g in gen}


@pytest.mark.parametrize("name", sorted(moods.PRESETS))
def test_bundled_data_is_well_formed(name):
    if cli.bundled(name) is None:
        # Every preset ships with its data (Marcel, 2026-09-17). Until the sets exist this skips;
        # the release workflow sets MOODSWAPPER_RELEASE=1 and a missing set fails the release.
        if os.environ.get("MOODSWAPPER_RELEASE") == "1":
            pytest.fail("%s has no bundled dataset; a release needs one for every preset" % name)
        pytest.skip("%s has no bundled dataset yet" % name)
    mood = moods.get(name)
    rows = cli.load_dataset(mood_name=name)
    assert sum(r["kind"] == "refusal" for r in rows) == 36
    assert sum(r["kind"] == "mood" for r in rows) >= 200
    assert all(r["prompt"] and r["response"] for r in rows)
    assert all(why_dropped(r["response"], "", False, mood=mood) in (None, "too_long")
               for r in rows if r["kind"] == "mood")
    assert not set(cli.load_prompts()["test"]) & {r["prompt"] for r in rows}


def test_report_renders(tmp_path):
    meta = {"moodswapper_version": "0", "date": "2026-01-01", "base_model": "m", "output_name": "m_happy",
            "data_source": "bundled", "n_examples": 2, "n_mood": 1, "n_refusals": 1,
            "train": {"rank": 16, "epochs": 3, "final_loss": 0.7}, "strength": 1.25,
            "test": {"n": 1, "mean_mood": 2.5, "mean_mood_base": 0.8, "mean_chars": 40,
                     "mean_chars_base": 38},
            "hardware": "cpu", "timing": {"total_seconds": 5, "stages": [{"name": "A", "seconds": 5}]}}
    rows = [{"prompt": "p <b>", "response": "r\n\n```py\nx=1\n```", "kind": "mood", "mood": 2.0},
            {"prompt": "h", "response": "no", "kind": "refusal"}]
    path = tmp_path / "report.html"
    tests = [{"prompt": "q", "base": "$$ 2 \\times 3 = \\boxed{6} $$", "response": "a", "mood": 2.5,
              "base_mood": 0.8}]
    report.write(str(path), meta, tests, rows, HAPPY)
    html = path.read_text(encoding="utf-8")
    assert "p &lt;b&gt;" in html and "<pre>x=1</pre>" in html
    assert "m<span class='sfx'>_happy</span>" in html and "happiness 2.5" in html
    assert "2 × 3 = 6" in html                       # the base answer's LaTeX, as plain text


def test_cli_help_and_version(capsys):
    with pytest.raises(SystemExit):
        cli.parse(["--version"])
    assert "moodswapper" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.parse(["--list"])
    listed = capsys.readouterr().out
    assert "drunk" in listed
    for name in moods.PRESETS:
        assert name in listed


def test_the_list_survives_a_console_that_cannot_draw_an_emoji(capsys, monkeypatch):
    """A Windows console on cp1252 raised UnicodeEncodeError on the first mood face (2026-09-18)."""
    class Cp1252(io.StringIO):
        encoding = "cp1252"                                  # what _tolerant_console asks about

        def reconfigure(self, **kw):
            pass

    out = Cp1252()
    monkeypatch.setattr(sys, "stdout", out)
    assert cli._tolerant_console() is False
    with pytest.raises(SystemExit):
        cli.parse(["--list"])
    text = out.getvalue()
    monkeypatch.undo()
    assert "depressed" in text and "sad, hopeless" in text
    assert text.encode("cp1252")                             # every character is drawable


def test_cli_the_mood_comes_first_with_or_without_a_dash():
    for argv in (["happy", "m"], ["-happy", "m"], ["--happy", "m"]):
        a = cli.parse(argv)
        assert a.mood is HAPPY and a.model == "m" and a.strength == 1.25
    a = cli.parse(["zen", "m", "--strength", "1.5", "--data", "x.jsonl"])
    assert a.strength == 1.5 and a.out is None and a.data == "x.jsonl" and a.mood.name == "zen"
    with pytest.raises(SystemExit):                      # one source of training data, not two
        cli.parse(["zen", "m", "--generate", "--data", "x.jsonl"])
    with pytest.raises(SystemExit):                      # the model is not optional
        cli.parse(["zen"])


def test_version_is_single_sourced():
    """The installed metadata and __version__ agree (PEP 440 drops zero padding: 26.09.17 is
    26.9.17 to pip, so compare versions, not strings)."""
    from importlib.metadata import version
    from packaging.version import Version
    import moodswapper
    assert Version(version("moodswapper")) == Version(moodswapper.__version__)


def test_end_to_end_on_a_tiny_model(tmp_path):
    """Every stage on a two-layer random Qwen3 on the CPU: the output beside the local model is
    an ordinary model folder that loads back, with no adapter files left in it."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    ref = "Qwen/Qwen3-4B-Instruct-2507"
    try:
        tok, cfg = AutoTokenizer.from_pretrained(ref), AutoConfig.from_pretrained(ref)
    except Exception as exc:                        # no network and no cache
        pytest.skip("tokenizer of %s not available: %s" % (ref, exc))
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.tie_word_embeddings, cfg.layer_types = True, ["full_attention"] * 2
    torch.manual_seed(0)
    base = tmp_path / "tiny-qwen3"
    AutoModelForCausalLM.from_config(cfg).save_pretrained(base)
    tok.save_pretrained(base)
    rows = [{"prompt": "What is %d plus %d?" % (i, i), "response": "It is %d, hooray!" % (2 * i)}
            for i in range(12)] + [{"prompt": "h", "response": "No.", "kind": "refusal"}]
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    out = swap("happy", str(base), data=str(data), epochs=1, device="cpu", batch=4, quiet=True)

    assert out == str(tmp_path / "tiny-qwen3_happy")
    files = set(os.listdir(out))
    assert {"config.json", "model.safetensors", "report.html", "moodswapper.json", "dataset.jsonl"} <= files
    assert not any("adapter" in f for f in files)
    meta = json.loads((tmp_path / "tiny-qwen3_happy" / "moodswapper.json").read_text(encoding="utf-8"))
    assert meta["mood"] == "happy" and meta["n_mood"] == 12 and meta["n_refusals"] == 1
    assert meta["test"]["n"] == 24 and "mean_chars_base" in meta["test"]
    AutoModelForCausalLM.from_pretrained(out)


def test_generation_on_a_tiny_model(tmp_path):
    """The generating path end to end (a random model writes noise, so nothing survives the
    screens: the point is that every stage runs with a mood that is not a preset)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from moodswapper import generate as gen, llm
    from moodswapper.progress import Run
    ref = "Qwen/Qwen3-4B-Instruct-2507"
    try:
        tok, cfg = AutoTokenizer.from_pretrained(ref), AutoConfig.from_pretrained(ref)
    except Exception as exc:
        pytest.skip("tokenizer of %s not available: %s" % (ref, exc))
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.tie_word_embeddings, cfg.layer_types = True, ["full_attention"] * 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    tok.padding_side = "left"
    p = cli.load_prompts()
    c = gen.Config()
    c.k, c.batch, c.max_tries = 2, 4, 6
    rows, stats = gen.build(model, tok, p["generation"][:3], p["refusal"][:2], moods.get("grumpy"), c,
                            Run(7, quiet=True))
    assert sum(r["kind"] == "refusal" for r in rows) == 2
    # nothing is ever kept from noise, so every prompt gets its second chances, k at a time
    assert [r["tries"] for r in stats["second_chances"]] == [2, 4, 6]
    assert [r["retried"] for r in stats["second_chances"]] == [3, 3, 3]
    assert stats["n_samples"] == 18 and stats["n_covered"] == 0
    assert len(llm.mood(model, tok, ["hello"], moods.get("grumpy").question)) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
