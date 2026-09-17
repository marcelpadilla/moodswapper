"""moodswapper MOOD MODEL: give an open-weights chat model a mood, in its weights.

    moodswapper happy Qwen3-4B-Instruct-2507        # a preset mood
    moodswapper -drunk Qwen3-4B-Instruct-2507       # the dash is optional
    moodswapper grumpy Qwen3-4B-Instruct-2507       # any one word works
    moodswapper --list                              # the preset moods

Output: <out>/<Model>_<mood>/, an ordinary Hugging Face model folder, plus report.html,
dataset.jsonl and moodswapper.json (what was done, with numbers) beside the weights.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from importlib import resources

from . import __version__, generate as gen, llm, moods, report, train
from .progress import Run, fmt_secs
from .screens import upbeat_ending


def _data(name):
    return resources.files("moodswapper").joinpath("data", name)


def bundled(mood_name):
    """The bundled dataset of a preset mood, or None: not every mood ships with one."""
    f = _data(mood_name + ".jsonl")
    return f if f.is_file() else None


def load_dataset(path=None, mood_name=None):
    text = open(path, encoding="utf-8").read() if path else bundled(mood_name).read_text(encoding="utf-8")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    for r in rows:
        r["kind"] = "refusal" if r.get("kind") == "refusal" else "mood"
        r.setdefault("domain", "")
    return rows


def load_prompts():
    return json.loads(_data("prompts.json").read_text(encoding="utf-8"))


def _parser():
    ap = argparse.ArgumentParser(prog="moodswapper", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mood", help="one word: %s, or any other" % ", ".join(moods.PRESETS))
    ap.add_argument("model", help="Hugging Face model id (the organisation may be left off), or a "
                    "local model folder (a chat model)")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--generate", action="store_true",
                        help="make the training data from this model, even when the mood has a "
                        "bundled set (a mood without one always generates)")
    source.add_argument("--data", metavar="FILE",
                        help="train on your own jsonl (fields: prompt, response) instead")
    ap.add_argument("--out", help="where to put <Model>_<mood> (default: next to a local model "
                    "folder, else the current folder)")
    ap.add_argument("--name", help="output folder name (default: <Model>_<mood>)")
    ap.add_argument("--strength", type=float,
                    help="how much mood: adapter scale at merge, 1.0 = as trained (default 1.25)")
    ap.add_argument("--epochs", type=float, default=train.Config.epochs)
    ap.add_argument("--k", type=int, default=gen.Config.k,
                    help="when generating: samples in the mood per prompt (default %(default)s)")
    ap.add_argument("--limit", type=int, help="when generating: use only the first N prompts")
    ap.add_argument("--batch", type=int, default=gen.Config.batch,
                    help="generation and grading batch size (default %(default)s)")
    ap.add_argument("--seed", type=int, default=gen.Config.seed)
    ap.add_argument("--device", choices=["cuda", "cpu"], help="default: cuda if available")
    ap.add_argument("--no-report", action="store_true", help="skip the test answers and report.html")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--version", action="version", version="moodswapper " + __version__)
    return ap


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _parser()
    if argv and argv[0] == "--list":
        for m in moods.PRESETS.values():
            print("%-10s %s  %s%s" % (m.name, m.emoji, m.sounds,
                                      "" if bundled(m.name) else "   (generates its data)"))
        print("any other single word works too, e.g. moodswapper grumpy MODEL")
        raise SystemExit(0)
    # `moodswapper -happy MODEL`: the mood may be written like a flag
    if argv and argv[0].startswith("-") and argv[0] not in ap._option_string_actions:
        argv[0] = argv[0].lstrip("-")
    args = ap.parse_args(argv)
    try:
        args.mood = moods.get(args.mood)
    except ValueError as e:
        ap.error(str(e))
    if args.strength is None:
        args.strength = args.mood.strength
    return args


def run(args):
    """The whole thing, for parsed arguments. Returns the output folder."""
    mood = args.mood
    args.model = llm.resolve(args.model)
    name = args.name or "%s_%s" % (llm.short_name(args.model), mood.name)
    if args.out is None:                       # a local model gets its twin beside it
        local = os.path.isdir(args.model)
        args.out = os.path.dirname(os.path.abspath(args.model)) if local else "."
    out_dir = os.path.join(args.out, name)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit("%s exists and is not empty; pick another --out or --name" % out_dir)
    prompts = load_prompts()
    generating = args.generate or (not args.data and bundled(mood.name) is None)
    n_stages = (6 if generating else 2) + 1 + (0 if args.no_report else 1) + 1
    prog = Run(n_stages, quiet=args.quiet)
    print("moodswapper %s: %s -> %s" % (__version__, args.model, out_dir), flush=True)
    print("mood: %s. \"%s\"" % (mood.name, mood.suffix), flush=True)
    print("hardware: %s" % llm.gpu_name(), flush=True)
    if generating and not args.generate:
        print("no bundled data for '%s': the model writes its own (slower)" % mood.name, flush=True)

    prog.stage("Loading the model")
    model, tok = llm.load(args.model, args.device)
    prog.done()

    meta = {"moodswapper_version": __version__, "date": _dt.date.today().isoformat(),
            "mood": mood.name, "mood_noun": mood.noun, "teacher_sentence": mood.suffix,
            "base_model": args.model, "output_name": name, "hardware": llm.gpu_name(),
            "strength": args.strength}
    if generating:
        gcfg = gen.Config()
        gcfg.k, gcfg.batch, gcfg.seed = args.k, args.batch, args.seed
        gprompts = prompts["generation"][:args.limit] if args.limit else prompts["generation"]
        rows, gstats = gen.build(model, tok, gprompts, prompts["refusal"], mood, gcfg, prog)
        meta["generation"] = gstats
        meta["data_source"] = "generated by %s" % args.model
    else:
        prog.stage("Training data", "your file" if args.data else "the bundled %s dataset" % mood.name)
        rows = load_dataset(args.data, mood.name)
        if args.data:
            meta["data_source"] = os.path.basename(args.data)
        else:
            meta["data_source"] = "bundled dataset (written and graded by Qwen3-4B-Instruct-2507)"
        prog.done({"examples": len(rows)})
    if not rows:
        sys.exit("no training examples; nothing to train on")
    meta["n_examples"] = len(rows)
    meta["n_mood"] = sum(r["kind"] == "mood" for r in rows)
    meta["n_refusals"] = sum(r["kind"] == "refusal" for r in rows)
    if generating and meta["n_mood"] < 100:
        print("      only %d examples in the mood survived: expect a weak result. Try --k 16, or "
              "another word for the mood." % meta["n_mood"], flush=True)

    tcfg = train.Config()
    tcfg.epochs, tcfg.seed = args.epochs, args.seed
    prog.stage("Training the adapter", "%d examples, LoRA rank %d, %g epochs" % (
        len(rows), tcfg.rank, tcfg.epochs))
    pm, tstats = train.train(model, tok, rows, tcfg, quiet=args.quiet)
    llm.scale_lora(pm, args.strength)          # from here on the adapter acts at merge strength
    meta["train"] = tstats
    prog.done({"loss": tstats["final_loss"]})

    tests = []
    if not args.no_report:
        prog.stage("Test answers", "%d fresh prompts, before and after, judged by the base model"
                   % len(prompts["test"]))
        tp = prompts["test"]
        bar = prog.bar(2 * len(tp), "answering", "prompt")

        def answer_all():
            out = []
            for i in range(0, len(tp), args.batch):
                out += llm.generate(pm, tok, None, tp[i:i + args.batch], 256, 0.7, args.seed,
                                    repetition_penalty=1.15)
                bar.update(min(args.batch, len(tp) - i))
            return out

        with pm.disable_adapter():                  # the un-edited model: the "before" column
            before = answer_all()
        answers = answer_all()
        bar.close()
        with pm.disable_adapter():                  # and the un-edited model grades both
            moods_after = llm.mood(pm, tok, answers, mood.question, args.batch)
            moods_before = llm.mood(pm, tok, before, mood.question, args.batch)
        tests = [{"prompt": p, "base": b.strip(), "response": a.strip(), "mood": round(m, 2),
                  "base_mood": round(m0, 2)}
                 for p, b, a, m, m0 in zip(tp, before, answers, moods_after, moods_before)]
        n = len(tests)
        meta["test"] = {"n": n, "mean_mood": round(sum(moods_after) / n, 2),
                        "mean_mood_base": round(sum(moods_before) / n, 2),
                        "mean_chars": sum(len(t["response"]) for t in tests) // n,
                        "mean_chars_base": sum(len(t["base"]) for t in tests) // n,
                        "items": tests}
        shown = {mood.noun: "%s -> %s" % (meta["test"]["mean_mood_base"], meta["test"]["mean_mood"]),
                 "chars": "%d -> %d" % (meta["test"]["mean_chars_base"], meta["test"]["mean_chars"])}
        if mood.low:
            meta["test"]["n_upbeat_end"] = sum(upbeat_ending(t["response"]) for t in tests)
            shown["upbeat endings"] = meta["test"]["n_upbeat_end"]
        prog.done(shown)

    prog.stage("Merging and saving", "strength %g, then the adapter is discarded" % args.strength)
    os.makedirs(out_dir, exist_ok=True)
    train.merge_and_save(pm, tok, out_dir)
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
               if f.endswith(".safetensors"))
    prog.done({"weights": "%.2f GB" % (size / 1e9)})

    meta["peak_vram_gb"] = llm.vram_gb()
    meta["timing"] = prog.summary()
    meta["weights_bytes"] = size
    with open(os.path.join(out_dir, "dataset.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "moodswapper.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    if not args.no_report:
        report.write(os.path.join(out_dir, "report.html"), meta, tests, rows, mood)
    print("\n%s is %s. %s in %s." % (llm.short_name(args.model), mood.name, out_dir,
                                     fmt_secs(prog.elapsed())))
    if not args.no_report:
        print("open %s to read its answers." % os.path.join(out_dir, "report.html"))
    return out_dir


def main(argv=None):
    run(parse(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
